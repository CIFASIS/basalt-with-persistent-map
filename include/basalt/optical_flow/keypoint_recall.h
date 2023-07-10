/**
  License
*/
#pragma once

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/vi_estimator/landmark_database.h>


namespace basalt {


class KeypointRecall {
 public:

  using Scalar = float;

  using Ptr = std::shared_ptr<KeypointRecall>;
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
  using SE3 = Sophus::SE3<Scalar>;

  KeypointRecall(const VioConfig& config, const basalt::Calibration<double>& calib)
      : calib_(calib.template cast<Scalar>()) {
    input_matching_queue.set_capacity(10);
    this->config_ = config;
}

  tbb::concurrent_bounded_queue<OpticalFlowResult::Ptr> input_matching_queue;
  tbb::concurrent_bounded_queue<OpticalFlowResult::Ptr>* output_matching_queue{};

  void initialize();

  void processFrame(OpticalFlowResult::Ptr& frame);

  void getProjectedLandmarks(OpticalFlowResult::Ptr& frame, size_t j, Eigen::aligned_unordered_map<LandmarkId, Landmark<float>>& landmarks);

  virtual ~KeypointRecall() { maybeJoin(); }

  inline void maybeJoin() {
    if (processing_thread_) {
      processing_thread_->join();
      processing_thread_.reset();
    }
  }

  // Macro defined in the Eigen library, which is a C++ library for linear algebra.
  // This macro is used to enable memory alignment for instances of a class that contain Eigen types, such as matrices or vectors.
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // TODO: how to define lmdb for doubles without template this class?
  LandmarkDatabase<Scalar>& lmdb_ = LandmarkDatabase<Scalar>::getMap();

 private:

  std::shared_ptr<std::thread> processing_thread_;
  VioConfig config_;
  const Calibration<Scalar> calib_;
  int num_matches_ = 0;

  // timing and stats
  // ExecutionStats stats_all_;
  // ExecutionStats stats_sums_;
};
}  // namespace basalt
