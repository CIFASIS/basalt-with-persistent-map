/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <memory>

#include <sophus/se2.hpp>
#include "basalt/imu/imu_types.h"
#include "basalt/imu/preintegration.h"
#include "basalt/utils/common_types.h"
#include "basalt/utils/imu_types.h"
#include "basalt/vi_estimator/landmark_database.h"
#include "sophus/se3.hpp"

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/optical_flow/patch.h>

#include <basalt/utils/keypoints.h>

namespace basalt {

/// Unlike PatchOpticalFlow, FrameToFrameOpticalFlow always tracks patches
/// against the previous frame, not the initial frame where a track was created.
/// While it might cause more drift of the patch location, it leads to longer
/// tracks in practice.
template <typename Scalar, template <typename> typename Pattern>
class FrameToFrameOpticalFlow : public OpticalFlowTyped<Scalar, Pattern> {
 public:
  using Vector3d = Eigen::Matrix<double, 3, 1>;

  using typename OpticalFlowTyped<Scalar, Pattern>::PatchT;
  using typename OpticalFlowTyped<Scalar, Pattern>::Vector2;
  using typename OpticalFlowTyped<Scalar, Pattern>::Matrix2;
  using typename OpticalFlowTyped<Scalar, Pattern>::Vector3;
  using typename OpticalFlowTyped<Scalar, Pattern>::Matrix3;
  using typename OpticalFlowTyped<Scalar, Pattern>::Vector4;
  using typename OpticalFlowTyped<Scalar, Pattern>::Matrix4;
  using typename OpticalFlowTyped<Scalar, Pattern>::SE2;
  using typename OpticalFlowTyped<Scalar, Pattern>::SE3;
  using OpticalFlowTyped<Scalar, Pattern>::getNumCams;
  using OpticalFlowTyped<Scalar, Pattern>::calib;
  using OpticalFlowTyped<Scalar, Pattern>::E;

  using OpticalFlowBase::config;
  using OpticalFlowBase::depth_guess;
  using OpticalFlowBase::first_state_arrived;
  using OpticalFlowBase::frame_counter;
  using OpticalFlowBase::input_depth_queue;
  using OpticalFlowBase::input_img_queue;
  using OpticalFlowBase::input_imu_queue;
  using OpticalFlowBase::input_state_queue;
  using OpticalFlowBase::last_keypoint_id;
  using OpticalFlowBase::latest_state;
  using OpticalFlowBase::old_pyramid;
  using OpticalFlowBase::output_queue;
  using OpticalFlowBase::patch_coord;
  using OpticalFlowBase::predicted_state;
  using OpticalFlowBase::processing_thread;
  using OpticalFlowBase::pyramid;
  using OpticalFlowBase::show_gui;
  using OpticalFlowBase::t_ns;
  using OpticalFlowBase::transforms;
  using OpticalFlowBase::matches_counter;

  FrameToFrameOpticalFlow(const VioConfig& conf, const Calibration<double>& cal)
      : OpticalFlowTyped<Scalar, Pattern>(conf, cal),
        accel_cov(cal.dicrete_time_accel_noise_std().array().square()),
        gyro_cov(cal.dicrete_time_gyro_noise_std().array().square()) {
    latest_state = std::make_shared<PoseVelBiasState<double>>();
    predicted_state = std::make_shared<PoseVelBiasState<double>>();
  }

  void processingLoop() override {
    using std::make_shared;
    OpticalFlowInput::Ptr img;

    while (true) {
      input_img_queue.pop(img);

      if (img == nullptr) {
        if (output_queue) output_queue->push(nullptr);
        break;
      }
      img->addTime("frames_received");

      while (input_depth_queue.try_pop(depth_guess)) continue;
      if (show_gui) img->depth_guess = depth_guess;

      if (!input_state_queue.empty()) {
        while (input_state_queue.try_pop(latest_state)) continue;  // Flush
        first_state_arrived = true;
      } else if (first_state_arrived) {
        latest_state = make_shared<PoseVelBiasState<double>>(*predicted_state);
      }

      if (first_state_arrived) {
        auto pim = processImu(img->t_ns);
        pim.predictState(*latest_state, constants::g, *predicted_state);
      }

      processFrame(img->t_ns, img);
    }
    showStats();
  }

  IntegratedImuMeasurement<double> processImu(int64_t curr_t_ns) {
    int64_t prev_t_ns = t_ns;
    Vector3d bg = latest_state->bias_gyro;
    Vector3d ba = latest_state->bias_accel;
    IntegratedImuMeasurement<double> pim{prev_t_ns, bg, ba};

    if (input_imu_queue.empty()) return pim;

    auto pop_imu = [&](ImuData<double>::Ptr& data) -> bool {
      input_imu_queue.pop(data);  // Blocking pop
      if (data == nullptr) return false;

      // Calibrate sample
      Vector3 a = calib.calib_accel_bias.getCalibrated(data->accel.cast<Scalar>());
      Vector3 g = calib.calib_gyro_bias.getCalibrated(data->gyro.cast<Scalar>());
      data->accel = a.template cast<double>();
      data->gyro = g.template cast<double>();
      return true;
    };

    typename ImuData<double>::Ptr data = nullptr;
    if (!pop_imu(data)) return pim;

    while (data->t_ns <= prev_t_ns) {
      if (!pop_imu(data)) return pim;
    }

    while (data->t_ns <= curr_t_ns) {
      pim.integrate(*data, accel_cov, gyro_cov);
      if (!pop_imu(data)) return pim;
    }

    // Pretend last IMU sample before "now" happened now
    if (pim.get_start_t_ns() + pim.get_dt_ns() < curr_t_ns) {
      data->t_ns = curr_t_ns;
      pim.integrate(*data, accel_cov, gyro_cov);
    }

    return pim;
  }

  void processFrame(int64_t curr_t_ns, OpticalFlowInput::Ptr& new_img_vec) {
    for (const auto& v : new_img_vec->img_data) {
      if (!v.img.get()) return;
    }

    const size_t num_cams = getNumCams();

    if (t_ns < 0) {
      t_ns = curr_t_ns;

      transforms.reset(new OpticalFlowResult);
      transforms->keypoints.resize(num_cams);
      transforms->tracking_guesses.resize(num_cams);
      transforms->matching_guesses.resize(num_cams);
      transforms->projections.resize(num_cams);
      transforms->recall_matches.resize(num_cams);
      transforms->new_detections.resize(num_cams);
      transforms->t_ns = t_ns;

      pyramid.reset(new std::vector<ManagedImagePyr<uint16_t>>);
      pyramid->resize(num_cams);

      tbb::parallel_for(tbb::blocked_range<size_t>(0, num_cams), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          pyramid->at(i).setFromImage(*new_img_vec->img_data[i].img, config.optical_flow_levels);
        }
      });

      transforms->input_images = new_img_vec;

      addPoints();
      filterPoints();
    } else {
      t_ns = curr_t_ns;

      old_pyramid = pyramid;

      pyramid.reset(new std::vector<ManagedImagePyr<uint16_t>>);
      pyramid->resize(num_cams);
      tbb::parallel_for(tbb::blocked_range<size_t>(0, num_cams), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          pyramid->at(i).setFromImage(*new_img_vec->img_data[i].img, config.optical_flow_levels);
        }
      });

      OpticalFlowResult::Ptr new_transforms;
      new_transforms.reset(new OpticalFlowResult);
      new_transforms->keypoints.resize(num_cams);
      new_transforms->tracking_guesses.resize(num_cams);
      new_transforms->matching_guesses.resize(num_cams);
      new_transforms->projections.resize(num_cams);
      new_transforms->recall_matches.resize(num_cams);
      new_transforms->new_detections.resize(num_cams);
      new_transforms->t_ns = t_ns;

      SE3 T_i1 = latest_state->T_w_i.template cast<Scalar>();
      SE3 T_i2 = predicted_state->T_w_i.template cast<Scalar>();
      for (size_t i = 0; i < num_cams; i++) {
        SE3 T_c1 = T_i1 * calib.T_i_c[i];
        SE3 T_c2 = T_i2 * calib.T_i_c[i];
        SE3 T_c1_c2 = T_c1.inverse() * T_c2;
        trackPoints(old_pyramid->at(i), pyramid->at(i),  //
                    transforms->keypoints[i], new_transforms->keypoints[i],
                    new_transforms->tracking_guesses[i],  //
                    new_img_vec->masks.at(i), new_img_vec->masks.at(i), T_c1_c2, i, i);
        opt_flow_counter_ += new_transforms->keypoints[i].size();
      }

      transforms = new_transforms;
      transforms->input_images = new_img_vec;

      recallPoints();
      addPoints();
      filterPoints();
    }

    if (output_queue && frame_counter % config.optical_flow_skip_frames == 0) {
      transforms->input_images->addTime("opticalflow_produced");
      output_queue->push(transforms);
    }

    for (size_t i = 0; i < num_cams; i++) points_counter_ += transforms->keypoints[i].size();
    frame_counter++;
  }

  void trackPoints(const ManagedImagePyr<uint16_t>& pyr_1, const ManagedImagePyr<uint16_t>& pyr_2,  //
                   const Keypoints& keypoint_map_1, Keypoints& keypoint_map_2, Poses& guesses,      //
                   const Masks& masks1, const Masks& masks2, const SE3& T_c1_c2, size_t cam1, size_t cam2) const {
    size_t num_points = keypoint_map_1.size();

    std::vector<KeypointId> ids;
    Eigen::aligned_vector<Keypoint> init_vec;

    ids.reserve(num_points);
    init_vec.reserve(num_points);

    for (const auto& [kpid, affine] : keypoint_map_1) {
      ids.push_back(kpid);
      init_vec.push_back(affine);
    }

    tbb::concurrent_unordered_map<KeypointId, Keypoint, std::hash<KeypointId>> result;
    tbb::concurrent_unordered_map<KeypointId, Eigen::AffineCompact2f, std::hash<KeypointId>> guesses_tbb;

    bool tracking = cam1 == cam2;
    bool matching = cam1 != cam2;
    MatchingGuessType guess_type = config.optical_flow_matching_guess_type;
    bool match_guess_uses_depth = guess_type != MatchingGuessType::SAME_PIXEL;
    const bool use_depth = tracking || (matching && match_guess_uses_depth);
    const double depth = depth_guess;

    auto compute_func = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const KeypointId id = ids[r];

        const Eigen::AffineCompact2f& transform_1 = init_vec[r].pose;
        Eigen::AffineCompact2f transform_2 = transform_1;

        auto t1 = transform_1.translation();
        auto t2 = transform_2.translation();

        if (masks1.inBounds(t1.x(), t1.y())) continue;

        bool valid = true;

        Eigen::Vector2f off{0, 0};

        if (use_depth) {
          Vector2 t2_guess;
          Scalar _;
          calib.projectBetweenCams(t1, depth, t2_guess, _, T_c1_c2, cam1, cam2);
          off = t2 - t2_guess;
        }

        t2 -= off;  // This modifies transform_2

        if (show_gui) {
          guesses_tbb[id] = transform_2;
        }

        valid = t2(0) >= 0 && t2(1) >= 0 && t2(0) < pyr_2.lvl(0).w && t2(1) < pyr_2.lvl(0).h;
        if (!valid) continue;

        valid = trackPoint(pyr_1, pyr_2, transform_1, transform_2);
        if (!valid) continue;

        if (masks2.inBounds(t2.x(), t2.y())) continue;

        Eigen::AffineCompact2f transform_1_recovered = transform_2;
        auto t1_recovered = transform_1_recovered.translation();

        t1_recovered += off;

        valid = trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered);
        if (!valid) continue;

        Scalar dist2 = (t1 - t1_recovered).squaredNorm();

        if (dist2 < config.optical_flow_max_recovered_dist2) {
          result[id].pose = transform_2;
          result[id].descriptor = init_vec[r].descriptor;
          result[id].tracked_by_opt_flow = true;
        }
      }
    };

    tbb::blocked_range<size_t> range(0, num_points);
    tbb::parallel_for(range, compute_func);

    keypoint_map_2.clear();
    keypoint_map_2.insert(result.begin(), result.end());
    guesses.clear();
    guesses.insert(guesses_tbb.begin(), guesses_tbb.end());
  }

  inline bool trackPoint(const ManagedImagePyr<uint16_t>& old_pyr, const ManagedImagePyr<uint16_t>& pyr,
                         const Eigen::AffineCompact2f& old_transform, Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    transform.linear().setIdentity();

    for (int level = config.optical_flow_levels; level >= 0 && patch_valid; level--) {
      const Scalar scale = 1 << level;

      transform.translation() /= scale;

      PatchT p(old_pyr.lvl(level), old_transform.translation() / scale);

      patch_valid &= p.valid;
      if (patch_valid) {
        // Perform tracking on current level
        patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);
      }

      transform.translation() *= scale;
    }

    transform.linear() = old_transform.linear() * transform.linear();

    return patch_valid;
  }

  inline bool trackPointAtLevel(const Image<const uint16_t>& img_2, const PatchT& dp,
                                Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    for (int iteration = 0; patch_valid && iteration < config.optical_flow_max_iterations; iteration++) {
      typename PatchT::VectorP res;

      typename PatchT::Matrix2P transformed_pat = transform.linear().matrix() * PatchT::pattern2;
      transformed_pat.colwise() += transform.translation();

      patch_valid &= dp.residual(img_2, transformed_pat, res);

      if (patch_valid) {
        const Vector3 inc = -dp.H_se2_inv_J_se2_T * res;

        // avoid NaN in increment (leads to SE2::exp crashing)
        patch_valid &= inc.array().isFinite().all();

        // avoid very large increment
        patch_valid &= inc.template lpNorm<Eigen::Infinity>() < 1e6;

        if (patch_valid) {
          transform *= SE2::exp(inc).matrix();

          const int filter_margin = 2;

          patch_valid &= img_2.InBounds(transform.translation(), filter_margin);
        }
      }
    }

    return patch_valid;
  }

  /**
   * @brief Project the landmarks into the current frame.
   * Returns the landmarks that are in the frame along with their corresponding 2D projections.
   *
   * @param[in] cam_id: The camera id of the current frame.
   * @param[out] landmarks: A reference where landmarks will be returned.
   * @param[out] projections: A reference where the landmark's projections will be returned.
  */
  void getProjectedLandmarks(size_t cam_id, Eigen::aligned_unordered_map<LandmarkId, Landmark<float>>& landmarks,  Eigen::aligned_unordered_map<LandmarkId, Vector2>& projections) {
    for (const auto& [lm_id, lm] : lmdb_.getLandmarks()) {

      // Skip landmarks that are already tracked by the current frame
      if (transforms->keypoints.at(cam_id).find(lm_id) != transforms->keypoints.at(cam_id).end()) continue;
      // Host camera
      size_t i = lm.host_kf_id.cam_id;

      // Unproject the direction vector
      Vector4 ci_xyzw = StereographicParam<Scalar>::unproject(lm.direction);
      ci_xyzw[3] = lm.inv_dist;

      // Get the transformation from the world to the host camera
      SE3 T_i_ci = calib.T_i_c[i];
      SE3 T_i0 = lmdb_.getFramePose(lm.host_kf_id.frame_id).template cast<Scalar>();
      SE3 T_w_ci = T_i0 * T_i_ci;

      Vector4 w_xyzw = T_w_ci * ci_xyzw;
      Vector3 w_xyz = w_xyzw.template head<3>() / w_xyzw[3];

      SE3 T_i1 = predicted_state->T_w_i.template cast<Scalar>();
      SE3 T_i_cj = calib.T_i_c[cam_id];
      SE3 T_cj = T_i1 * T_i_cj;
      Vector3 cj_xyz = T_cj.inverse() * w_xyz;
      Vector2 cj_uv;
      // Project the point to the new frame
      bool valid = calib.intrinsics[cam_id].project(cj_xyz, cj_uv);

      // Check if the point is in the bounds of the frame
      const basalt::Image<const uint16_t>& img_raw = pyramid->at(cam_id).lvl(0);
      bool in_bounds = cj_uv.x() >= 0 && cj_uv.x() < img_raw.w && cj_uv.y() >= 0 && cj_uv.y() < img_raw.h;
      if (valid && in_bounds) {
        landmarks[lm_id] = lm;
        projections[lm_id] = cj_uv;
        transforms->projections.at(cam_id).emplace_back(std::make_tuple(lm_id, cj_uv));
      }
    }
  }

  /**
   * @brief Given a 2D point calculates in which cell of the grid the point lies.
   *
   * @param[in] p: 2D point.
   *
   * @return The cell ID.
  */
  int getPointCell(Vector2& p) {
      const basalt::Image<const uint16_t>& img_raw = pyramid->at(0).lvl(0);

      // size of the cell side
      int size = config.optical_flow_detection_grid_size;

      // Determine the number of cells in each row and column.
      int cellsPerRow = static_cast<int>(img_raw.w / size);
      int cellsPerCol = static_cast<int>(img_raw.h / size);

      // Calculate the extra width and height due to the non-multiple of SIZE frame dimensions.
      // These extra dimensions will be split between the border cells, making the first and last
      // row and column of cells slightly bigger than the rest.
      float extraWidth = (img_raw.w - cellsPerRow * size) / 2;
      float extraHeight = (img_raw.h - cellsPerCol * size) / 2;

      // Calculate the adjusted coordinates of the point to account for the extra border cells.
      float new_x = p.x() - extraWidth > 0 ? p.x() - extraWidth : 0;
      float new_y = p.y() - extraHeight > 0 ? p.y() - extraHeight : 0;

      // Calculate the row and column index of the cell in which the point resides.
      // If the adjusted coordinates fall outside the last cell, they will be capped to the maximum
      // row and column index, ensuring the point is assigned to the last cell in such cases.
      int row = static_cast<int>(new_y / size) > cellsPerCol - 1 ? cellsPerCol - 1 : static_cast<int>(new_y / size);
      int col = static_cast<int>(new_x / size) > cellsPerRow - 1 ? cellsPerRow - 1 : static_cast<int>(new_x / size);
      return cellsPerRow * row + col;
  }

  /**
   * @brief returns the number of cells of the grid.
   *
   * @return The cell ID.
  */
  int getNumCells() {
    const basalt::Image<const uint16_t>& img_raw = pyramid->at(0).lvl(0);
    int size = config.optical_flow_detection_grid_size;
    int cellsPerRow = static_cast<int>(img_raw.w / size);
    int cellsPerCol = static_cast<int>(img_raw.h / size);
    return cellsPerRow * cellsPerCol;
  }

  /**
   *  @brief Function responsible for matching the new detections with the projections of the landmarks stored in the map.
   *
   *  Algorithm Steps:
   *  1. Detect keypoints in the new frame using FAST.
   *  2. Project the landmarks from the map into the new frame to obtain their projections.
   *  3. Group the landmarks' projections by cells and search for detected points in the nearest cells.
   *  4. Match the descriptors of newly detected points with the descriptor of each landmark.
   *  5. If a match is found, associate the detected keypoint with landmark ID and store the information.
   *
  */
  void recallPoints() {
    for (size_t cam_id = 0; cam_id < getNumCams(); cam_id++) {
      std::vector<KeypointId> new_points_index;
      std::vector<Descriptor> new_points_descriptors;
      std::vector<KeypointId> landmarks_ids;
      std::vector<Descriptor> landmarks_descriptors;

      std::vector<std::pair<int, int>> matches;

      // 1. Detect keypoints in the new frame using FAST.
      KeypointsData kd;
      Eigen::aligned_vector<Eigen::Vector2d> pts;
      detectKeypoints(pyramid->at(cam_id).lvl(0), kd, config.optical_flow_detection_grid_size,
                      config.recall_detection_num_points_cell, config.optical_flow_detection_min_threshold,
                      config.optical_flow_detection_max_threshold, transforms->input_images->masks.at(cam_id), pts);
      computeAngles(pyramid->at(cam_id).lvl(0), kd, true);
      computeDescriptors(pyramid->at(cam_id).lvl(0), kd);

      // 2. Project the landmarks from the map into the new frame to obtain their projections.
      Eigen::aligned_unordered_map<LandmarkId, Landmark<float>> proj_landmarks;
      Eigen::aligned_unordered_map<LandmarkId, Vector2> projections;
      getProjectedLandmarks(cam_id, proj_landmarks, projections);

      // 3. Group the landmark's projections by cells and search for detected points in the their cells.
      // TODO: this could be done in parallel
      for (int cell_id = 0; cell_id < getNumCells(); cell_id++) {

        new_points_index.clear();
        new_points_descriptors.clear();
        for (size_t i = 0; i < kd.corners.size(); i++) {
          Vector2 pos = kd.corners[i].cast<Scalar>();
          if (getPointCell(pos) == cell_id) {
            new_points_index.push_back(i);
            new_points_descriptors.push_back(kd.corner_descriptors[i]);
            transforms->new_detections.at(cam_id).emplace_back(pos);
          }
        }

        landmarks_ids.clear();
        landmarks_descriptors.clear();
        for (const auto& [lm_id, lm] : proj_landmarks) {
          Vector2 pos = projections.at(lm_id);
          if (getPointCell(pos) == cell_id) {
            landmarks_ids.push_back(lm_id);
            landmarks_descriptors.push_back(lm.descriptor);
          }
        }

        // 4. Match the descriptors of newly detected points with the descriptor of each landmark.
        matches.clear();
        matchDescriptors(new_points_descriptors, landmarks_descriptors, matches, config.mapper_max_hamming_distance, config.mapper_second_best_test_ratio);

        // 5. If a match is found, associate the detected keypoint with landmark ID and store the information.
        for (auto & match : matches) {
          size_t point_idx = new_points_index[match.first];
          size_t lm_id = landmarks_ids[match.second];

          auto transform = Eigen::AffineCompact2f::Identity();
          transform.translation() = kd.corners[point_idx].cast<Scalar>();
          transforms->keypoints.at(cam_id)[lm_id].pose = transform;
          transforms->keypoints.at(cam_id)[lm_id].descriptor = kd.corner_descriptors[point_idx];
          transforms->keypoints.at(cam_id)[lm_id].tracked_by_recall = true;
          std::tuple<int64_t, Vector2, Vector2> match_pair = std::make_tuple(lm_id, kd.corners[point_idx].cast<Scalar>(), projections.at(lm_id));
          transforms->recall_matches.at(cam_id).emplace_back(match_pair);
          matches_counter_++;
        }
      }
    }
  }

  Keypoints addPointsForCamera(size_t cam_id) {
    Eigen::aligned_vector<Eigen::Vector2d> pts;  // Current points
    for (const auto& [kpid, affine] : transforms->keypoints.at(cam_id)) {
      pts.emplace_back(affine.pose.translation().template cast<double>());
    }

    KeypointsData kd;  // Detected new points
    detectKeypoints(pyramid->at(cam_id).lvl(0), kd, config.optical_flow_detection_grid_size,
                    config.optical_flow_detection_num_points_cell, config.optical_flow_detection_min_threshold,
                    config.optical_flow_detection_max_threshold, transforms->input_images->masks.at(cam_id), pts);
    computeAngles(pyramid->at(cam_id).lvl(0), kd, true);
    computeDescriptors(pyramid->at(cam_id).lvl(0), kd);

    Keypoints new_kpts;
    for (size_t i = 0; i < kd.corners.size(); i++) {  // Set new points as keypoints
      auto transform = Eigen::AffineCompact2f::Identity();
      transform.translation() = kd.corners[i].cast<Scalar>();

      transforms->keypoints.at(cam_id)[last_keypoint_id].pose = transform;
      transforms->keypoints.at(cam_id)[last_keypoint_id].descriptor = kd.corner_descriptors[i];
      transforms->keypoints.at(cam_id)[last_keypoint_id].tracked_by_opt_flow = false;
      new_kpts[last_keypoint_id] = transforms->keypoints.at(cam_id)[last_keypoint_id];

      last_keypoint_id++;
    }

    return new_kpts;
  }

  Masks cam0OverlapCellsMasksForCam(size_t cam_id) {
    int C = config.optical_flow_detection_grid_size;  // cell size

    int w = transforms->input_images->img_data.at(cam_id).img->w;
    int h = transforms->input_images->img_data.at(cam_id).img->h;

    int x_start = (w % C) / 2;
    int y_start = (h % C) / 2;

    int x_stop = x_start + C * (w / C - 1);
    int y_stop = y_start + C * (h / C - 1);

    int x_first = x_start + C / 2;
    int y_first = y_start + C / 2;

    int x_last = x_stop + C / 2;
    int y_last = y_stop + C / 2;

    Masks masks;
    for (int y = y_first; y <= y_last; y += C) {
      for (int x = x_first; x <= x_last; x += C) {
        Vector2 ci_uv{x, y};
        Vector2 c0_uv;
        Scalar _;
        bool projected = calib.projectBetweenCams(ci_uv, depth_guess, c0_uv, _, cam_id, 0);
        bool in_bounds = c0_uv.x() >= 0 && c0_uv.x() < w && c0_uv.y() >= 0 && c0_uv.y() < h;
        bool valid = projected && in_bounds;
        if (valid) {
          Rect cell_mask(x - C / 2, y - C / 2, C, C);
          masks.masks.push_back(cell_mask);
        }
      }
    }
    return masks;
  }

  void addPoints() {
    Masks& ms0 = transforms->input_images->masks.at(0);
    Keypoints kpts0 = addPointsForCamera(0);

    for (size_t i = 1; i < getNumCams(); i++) {
      Masks& ms = transforms->input_images->masks.at(i);
      Poses& mgs = transforms->matching_guesses.at(i);

      // Match features on areas that overlap with cam0 using optical flow
      auto& pyr0 = pyramid->at(0);
      auto& pyri = pyramid->at(i);
      Keypoints kpts;
      SE3 T_c0_ci = calib.T_i_c[0].inverse() * calib.T_i_c[i];
      trackPoints(pyr0, pyri, kpts0, kpts, mgs, ms0, ms, T_c0_ci, 0, i);
      transforms->keypoints.at(i).insert(kpts.begin(), kpts.end());

      // Update masks and detect features on area not overlapping with cam0
      if (!config.optical_flow_detection_nonoverlap) continue;
      ms += cam0OverlapCellsMasksForCam(i);
      Keypoints kpts_no = addPointsForCamera(i);
    }
  }

  void filterPointsForCam(int cam_id) {
    std::set<KeypointId> kp_to_remove;

    std::vector<KeypointId> kpids;
    Eigen::aligned_vector<Eigen::Vector2f> proj0, proj1;

    for (const auto& [kpid, affine] : transforms->keypoints.at(cam_id)) {
      auto it = transforms->keypoints.at(0).find(kpid);

      if (it != transforms->keypoints.at(0).end()) {
        proj0.emplace_back(it->second.pose.translation());
        proj1.emplace_back(affine.pose.translation());
        kpids.emplace_back(kpid);
      }
    }

    Eigen::aligned_vector<Eigen::Vector4f> p3d0, p3d1;
    std::vector<bool> p3d0_success, p3d1_success;

    calib.intrinsics[0].unproject(proj0, p3d0, p3d0_success);
    calib.intrinsics[cam_id].unproject(proj1, p3d1, p3d1_success);

    for (size_t i = 0; i < p3d0_success.size(); i++) {
      if (p3d0_success[i] && p3d1_success[i]) {
        const double epipolar_error = std::abs(p3d0[i].transpose() * E[cam_id] * p3d1[i]);

        if (epipolar_error > config.optical_flow_epipolar_error) {
          kp_to_remove.emplace(kpids[i]);
        }
      } else {
        kp_to_remove.emplace(kpids[i]);
      }
    }

    for (int id : kp_to_remove) {
      transforms->keypoints.at(cam_id).erase(id);
    }
  }

  void filterPoints() {
    for (size_t i = 1; i < getNumCams(); i++) {
      filterPointsForCam(i);
    }
  }

  void showStats() {
    std::cout << std::endl;
    std::cout << "==== Front-end stats ====" << std::endl;
    std::cout << "Total Detected Points: " << points_counter_ << std::endl;
    std::cout << "OpticalFlow: " << opt_flow_counter_ << std::endl;
    std::cout << "Recall: " << matches_counter_ << std::endl;
    std::cout << "AVG Points per Frame: " << points_counter_ / frame_counter << std::endl;
    std::cout << std::endl;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  const Vector3d accel_cov;
  const Vector3d gyro_cov;
  LandmarkDatabase<Scalar>& lmdb_ = LandmarkDatabase<Scalar>::getMap();
  int points_counter_ = 0;
  int opt_flow_counter_ = 0;
  int matches_counter_ = 0;
};

}  // namespace basalt
