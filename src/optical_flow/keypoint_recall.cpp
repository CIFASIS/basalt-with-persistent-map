/**
  License
*/

#include <basalt/optical_flow/keypoint_recall.h>
#include <tbb/parallel_for.h>

namespace basalt {

/**
 * Initialize the keypoint matching process by creating a processing thread.
 * The processing thread continuously retrieves frames from the Optical Flow output queue,
 * matches keypoints in each frame, and pushes the processed frames into the output matching queue.
 *
 */
void KeypointRecall::initialize() {
  auto proc_func = [&] {
    std::cout << "Matching points..." << std::endl;
    OpticalFlowResult::Ptr curr_frame;
    while (true) {
      input_matching_queue.pop(curr_frame);
      if (curr_frame == nullptr) {
          output_matching_queue->push(nullptr);
          break;
        }
      // TODO: Implement the matching here
      processFrame(curr_frame);
      output_matching_queue->push(curr_frame);
    }
    std::cout << "Finished matching points. Total matches: " << num_matches_ << std::endl;
  };
  processing_thread_.reset(new std::thread(proc_func));
}

/**
 * Match keypoints in the current frame with keypoints in the landmark database.
 *
 * @param curr_frame A pointer to the current frame containing observations and descriptors.
 */
void KeypointRecall::processFrame(OpticalFlowResult::Ptr& curr_frame) {

  int NUM_CAMS = curr_frame->keypoints.size();
  for (int i=0; i < NUM_CAMS; i++) {
    std::vector<KeypointId> kp1;
    std::vector<KeypointId> kp2;
    std::vector<Descriptor> descr1;
    std::vector<Descriptor> descr2;

    for (const auto& [kpt_id, kpt] : curr_frame->keypoints.at(i)) {
      if (!kpt.tracked_by_opt_flow){
        kp1.push_back(kpt_id);
        descr1.push_back(kpt.descriptor);
      }
    }

    Eigen::aligned_unordered_map<LandmarkId, Landmark<float>> landmarks;
    Eigen::aligned_unordered_map<LandmarkId, Vec2> projections;
    getProjectedLandmarks(curr_frame, i, landmarks, projections);
    for (const auto& [lm_id, lm] : landmarks) {
      kp2.push_back(lm_id);
      descr2.push_back(lm.descriptor);
    }

    std::vector<std::pair<int, int>> matches;

    matchDescriptors(descr1, descr2, matches,
                      config_.mapper_max_hamming_distance,
                      config_.mapper_second_best_test_ratio);

    for (const auto& match: matches) {
      // If match: keypoint kp1[i] is the same as kp2[j] so change the kp_id
      KeypointId kp_id = kp1[match.first];
      KeypointId new_kp_id = kp2[match.second];
      // TODO: if we filter the klf matches this shouldn't be necessary
      if (new_kp_id != kp_id) {
        // TODO: check if this is necessary
        if (curr_frame->keypoints.at(i).count(kp_id) == 0 || curr_frame->keypoints.at(i).count(new_kp_id) > 0) {continue;}

        curr_frame->keypoints.at(i)[new_kp_id] = curr_frame->keypoints.at(i).at(kp_id);
        curr_frame->keypoints.at(i)[new_kp_id].tracked_by_recall = true;

        // return landmark projected pose
        curr_frame->projections[i].emplace_back(projections.at(new_kp_id));

        // return match poses
        std::tuple<Vec2, Vec2> match_pair;
        Vec2 kpt_pose = curr_frame->keypoints.at(i).at(new_kp_id).pose.translation();

        match_pair = std::make_tuple(kpt_pose, projections.at(new_kp_id));
        curr_frame->recall_matches.at(i).emplace_back(match_pair);
        curr_frame->keypoints.at(i).erase(kp_id);
        num_matches_++;
        std::cout << "New match at coords: " << curr_frame->keypoints.at(i).at(new_kp_id).pose.translation().transpose() << " camera " << i << " with keypoint id: " << new_kp_id << std::endl;
      }
    }
  }
}

void KeypointRecall::getProjectedLandmarks(OpticalFlowResult::Ptr& curr_frame, size_t j, Eigen::aligned_unordered_map<LandmarkId, Landmark<float>>& landmarks,  Eigen::aligned_unordered_map<LandmarkId, Vec2>& projections) {
  for (const auto& [lm_id, lm] : lmdb_.getLandmarks()) {

    // Host camera
    size_t i = lm.host_kf_id.cam_id;

    // Unproject the direction vector
    Vec4 ci_xyzw = StereographicParam<Scalar>::unproject(lm.direction);
    ci_xyzw[3] = lm.inv_dist;

    // Get the transformation from the world to the host camera
    SE3 T_i_ci = calib_.T_i_c[i];
    SE3 T_i0 = lmdb_.getFramePose(lm.host_kf_id.frame_id).template cast<Scalar>();
    SE3 T_w_ci = T_i0 * T_i_ci;

    Vec4 w_xyzw = T_w_ci * ci_xyzw;
    Vec3 w_xyz = w_xyzw.template head<3>() / w_xyzw[3];

    SE3 T_i1 = curr_frame->predicted_state->T_w_i.template cast<Scalar>();
    SE3 T_i_cj = calib_.T_i_c[j];
    SE3 T_cj = T_i1 * T_i_cj;
    Vec3 cj_xyz = T_cj.inverse() * w_xyz;
    Vec2 cj_uv;
    // Project the point to the new frame
    bool valid = calib_.intrinsics[j].project(cj_xyz, cj_uv);
    if (valid) {
      landmarks[lm_id] = lm;
      projections[lm_id] = cj_uv;
    }
  }
}
}  // namespace basalt
