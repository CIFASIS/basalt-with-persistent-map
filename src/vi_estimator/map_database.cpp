#include <basalt/vi_estimator/map_database.h>

namespace basalt {

MapDatabase::MapDatabase(const VioConfig& config, const Calibration<double>& calib)
    : out_vis_queue(nullptr) {
  this->config = config;
  this->calib = calib;
  this->map = LandmarkDatabase<float>("Persistent Map");
}

void MapDatabase::initialize() {
  auto read_func = [&]() {
    while (true) {
      auto keypoints_ptr = std::make_shared<std::vector<KeypointId>>();
      in_covi_req_queue.pop(keypoints_ptr);

      if (keypoints_ptr == nullptr) break;
      std::vector<KeypointId>& keypoints = *keypoints_ptr;

      std::unique_lock<std::mutex> lock(mutex);
      handleCovisibilityReq(keypoints);
    }
  };

  auto write_func = [&]() {
    basalt::MapStamp::Ptr map_stamp;
    while (true) {
      in_map_stamp_queue.pop(map_stamp);

      if (map_stamp == nullptr) {
        map.print();
        if (out_vis_queue) out_vis_queue->push(nullptr);
        break;
      }

      std::unique_lock<std::mutex> lock(mutex);
      map.mergeLMDB(map_stamp->lmdb, true);

      if (config.map_covisibility_criteria == MapCovisibilityCriteria::MAP_COV_STS) {
        std::set<TimeCamId> kfs_to_compute;
        for (auto const& [kf_id, _] : map_stamp->lmdb->getKeyframeObs()) kfs_to_compute.emplace(kf_id);
        computeSpatialDistributions(kfs_to_compute);
      }

      if (out_vis_queue) {
        map_visual_data = std::make_shared<MapDatabaseVisualizationData>();
        map_visual_data->t_ns = map_stamp->t_ns;
        computeMapVisualData();
        out_vis_queue->push(map_visual_data);
      }
    }
  };

  reading_thread.reset(new std::thread(read_func));
  writing_thread.reset(new std::thread(write_func));
}

void MapDatabase::get_map_points(Eigen::aligned_vector<Vec3d>& points, std::vector<int>& ids) {
  points.clear();
  ids.clear();

  for (const auto& tcid_host : map.getHostKfs()) {
    int64_t id = tcid_host.frame_id;
    Sophus::SE3d T_w_i = map.getKeyframePose(id).cast<double>();

    const Sophus::SE3d& T_i_c = calib.T_i_c[tcid_host.cam_id];
    Mat4d T_w_c = (T_w_i * T_i_c).matrix();

    for (const auto& [lm_id, lm_pos] : map.getLandmarksForHostWithIds(tcid_host)) {
      Vec4d pt_cam = StereographicParam<double>::unproject(lm_pos->direction.cast<double>());
      pt_cam[3] = lm_pos->inv_dist;

      Vec4d pt_w = T_w_c * pt_cam;

      points.emplace_back((pt_w.template head<3>() / pt_w[3]).template cast<double>());
      ids.emplace_back(lm_id);
    }
  }
}

Eigen::aligned_map<LandmarkId, Eigen::Matrix<double, 3, 1>> MapDatabase::get_landmarks_3d_pos(
    std::set<LandmarkId> landmarks) {
  Eigen::aligned_map<LandmarkId, Vec3d> landmarks_3d{};

  for (const auto lm_id : landmarks) {
    auto lm = map.getLandmark(lm_id);
    int64_t frame_id = lm.host_kf_id.frame_id;
    Sophus::SE3d T_w_i = map.getKeyframePose(frame_id).cast<double>();

    const Sophus::SE3d& T_i_c = calib.T_i_c[lm.host_kf_id.cam_id];
    Mat4d T_w_c = (T_w_i * T_i_c).matrix();

    Vec4d pt_cam = StereographicParam<double>::unproject(lm.direction.cast<double>());
    pt_cam[3] = lm.inv_dist;

    Vec4d pt_w = T_w_c * pt_cam;

    landmarks_3d.emplace(lm_id, (pt_w.template head<3>() / pt_w[3]).template cast<double>());
  }

  return landmarks_3d;
}

void MapDatabase::computeMapVisualData() {
  // show landmarks
  get_map_points(map_visual_data->landmarks, map_visual_data->landmarks_ids);

  // show keyframes
  for (const auto& [frame_id, pose] : map.getKeyframes()) {
    map_visual_data->keyframe_idx[frame_id] = map.getKeyframeIndex(frame_id);
    map_visual_data->keyframe_poses[frame_id] = pose.template cast<double>();
  }

  // show covisibility
  for (const auto& [tcid_h, target_map] : map.getObservations()) {
    for (const auto& [tcid_t, obs] : target_map) {
      Eigen::Vector3d p1 = map.getKeyframePose(tcid_h.frame_id).template cast<double>().translation();
      Eigen::Vector3d p2 = map.getKeyframePose(tcid_t.frame_id).template cast<double>().translation();
      map_visual_data->covisibility.emplace_back(p1);
      map_visual_data->covisibility.emplace_back(p2);
    }
  }

  // Show observations
  for (const auto& [tcid, obs] : map.getKeyframeObs()) {
    Eigen::Vector3d kf_pos = map.getKeyframePose(tcid.frame_id).template cast<double>().translation();
    auto landmarks_3d = get_landmarks_3d_pos(obs);
    for (const auto& lm_id : obs) {
      map_visual_data->observations[lm_id].emplace_back(kf_pos);
      map_visual_data->observations[lm_id].emplace_back(landmarks_3d[lm_id]);
    }
  }
}

void MapDatabase::handleCovisibilityReq(const std::vector<size_t>& curr_kpts) {
  LandmarkDatabase<Scalar>::Ptr covisible_submap{};

  if (config.map_covisibility_criteria == MapCovisibilityCriteria::MAP_COV_DEFAULT) {
    covisible_submap = std::make_shared<LandmarkDatabase<Scalar>>("Covisible Submap");
    map.getCovisibilityMap(covisible_submap);
  }
  else if (config.map_covisibility_criteria == MapCovisibilityCriteria::MAP_COV_STS) {
    computeSTSMap(curr_kpts);
    covisible_submap = std::make_shared<LandmarkDatabase<Scalar>>(*sts_map);
  }
  else {
    BASALT_LOG_FATAL("Unexpected covisibility criteria");
  }
  if (out_covi_res_queue) out_covi_res_queue->push(covisible_submap);
}

void MapDatabase::computeSpatialDistributions(const std::set<TimeCamId>& kfs_to_compute) {
  for (const auto& [kf_id, obs] : map.getKeyframeObs()) {
    if (kfs_to_compute.count(kf_id) == 0) continue;

    auto landmarks_3d = get_landmarks_3d_pos(obs);
    std::vector<Vec3d> points;
    points.reserve(landmarks_3d.size());
    for (const auto& [lm_id, lm_3d] : landmarks_3d) points.emplace_back(lm_3d);
    keyframes_sdc[kf_id] = SpatialDistributionCube<double>(points);
  }
}

void MapDatabase::computeSTSMap(const std::vector<size_t>& curr_kpts) {
  if (map.numKeyframes() == 0) return;

  SpatialDistributionCube<double> current_sdc;

  if (config.map_sts_use_last_frame) {
    std::set<size_t> curr_lms;
    for (const auto& kpid : curr_kpts) {
      if (map.landmarkExists(kpid)) curr_lms.insert(kpid);
    }
    std::vector<Vec3d> points;
    for (const auto& [lm_id, lm_3d] : get_landmarks_3d_pos(curr_lms)) points.emplace_back(lm_3d);
    current_sdc = SpatialDistributionCube<double>(points);
  } else {
    auto last_keyframe = map.getLastKeyframe();
    current_sdc = keyframes_sdc[last_keyframe];
  }

  std::set<TimeCamId> candidate_kfs;
  for (const auto& [kf_id, sdc] : keyframes_sdc) {
    if (current_sdc.hasOverlap(keyframes_sdc[kf_id])) candidate_kfs.insert(kf_id);
    if (static_cast<int>(candidate_kfs.size()) >= config.map_sts_max_size) break;
  }

  map.getSubmap(candidate_kfs, sts_map);
}

const std::map<std::string, double> MapDatabase::getStats(){
  std::map<std::string, double> stats{};
  stats["num_kfs"] = map.numKeyframes();
  stats["num_lms"] = map.numLandmarks();
  stats["num_obs"] = map.numObservations();
  return stats;
}

}  // namespace basalt
