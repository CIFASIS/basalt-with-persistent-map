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

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>

#include <fmt/format.h>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

#include <basalt/io/dataset_io.h>
#include <basalt/io/marg_data_io.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/utils/assert.h>
#include <basalt/vi_estimator/vio_estimator.h>
#include <basalt/calibration/calibration.hpp>

#include <basalt/serialization/headers_serialization.h>

#include <basalt/utils/keypoints.h>
#include <basalt/utils/system_utils.h>
#include <basalt/utils/vio_config.h>
#include <basalt/utils/vis_utils.h>
#include <basalt/utils/format.hpp>
#include <basalt/utils/time_utils.hpp>

#include <basalt/vi_estimator/nfr_mapper.h>

// enable the "..."_format(...) string literal
using namespace basalt::literals;
using namespace basalt;
using namespace Eigen;

// GUI functions
void draw_image_overlay(pangolin::View& v, size_t cam_id);
void draw_scene(pangolin::View& view);
void load_data(const std::string& calib_path);
bool next_step();
bool prev_step();
void draw_plots();
void alignButton();
void alignDeviceButton();
void saveTrajectoryButton();
// Map GUI functions
void initializeMapper();
void computeEdgeVis();
void loadMapData();
void showMap();
void draw_mapper_image_overlay(pangolin::View& v, size_t view_id);
void draw_map_scene();

// Pangolin variables
constexpr int UI_WIDTH = 200;

using Button = pangolin::Var<std::function<void(void)>>;

pangolin::DataLog imu_data_log, vio_data_log, error_data_log;
pangolin::Plotter* plotter;

pangolin::Var<int> show_frame("ui.show_frame", 0, 0, 1500);

pangolin::Var<bool> show_flow("ui.show_flow", false, false, true);
pangolin::Var<bool> show_tracking_guess("ui.show_tracking_guess", false, false, true);
pangolin::Var<bool> show_matching_guess("ui.show_matching_guess", false, false, true);
pangolin::Var<bool> show_obs("ui.show_obs", true, false, true);
pangolin::Var<bool> show_ids("ui.show_ids", false, false, true);
pangolin::Var<bool> show_depth{"ui.show_depth", false, false, true};

pangolin::Var<bool> show_grid{"ui.show_grid", false, false, true};
pangolin::Var<bool> show_cam0_proj{"ui.show_cam0_proj", false, false, true};
pangolin::Var<bool> show_masks{"ui.show_masks", false, false, true};

pangolin::Var<bool> show_guesses{"ui.Show matching guesses", false, false, true};
pangolin::Var<bool> show_same_pixel_guess{"ui.SAME_PIXEL", true, false, true};
pangolin::Var<bool> show_reproj_avg_depth_guess{"ui.REPROJ_AVG_DEPTH", true, false, true};
pangolin::Var<bool> show_reproj_fix_depth_guess{"ui.REPROJ_FIX_DEPTH", true, false, true};
pangolin::Var<double> fixed_depth{"ui.FIX_DEPTH", 2, 0, 3};
pangolin::Var<bool> show_active_guess{"ui.Active Guess", true, false, true};

pangolin::Var<double> depth_guess{"ui.depth_guess", 2, pangolin::META_FLAG_READONLY};

pangolin::Var<bool> show_est_pos("ui.show_est_pos", true, false, true);
pangolin::Var<bool> show_est_vel("ui.show_est_vel", false, false, true);
pangolin::Var<bool> show_est_bg("ui.show_est_bg", false, false, true);
pangolin::Var<bool> show_est_ba("ui.show_est_ba", false, false, true);

pangolin::Var<bool> show_gt("ui.show_gt", true, false, true);

Button next_step_btn("ui.next_step", &next_step);
Button prev_step_btn("ui.prev_step", &prev_step);

pangolin::Var<bool> continue_btn("ui.continue", false, false, true);
pangolin::Var<bool> continue_fast("ui.continue_fast", true, false, true);

Button align_se3_btn("ui.align_se3", &alignButton);

pangolin::Var<bool> euroc_fmt("ui.euroc_fmt", true, false, true);
pangolin::Var<bool> tum_rgbd_fmt("ui.tum_rgbd_fmt", false, false, true);
pangolin::Var<bool> kitti_fmt("ui.kitti_fmt", false, false, true);
pangolin::Var<bool> save_groundtruth("ui.save_groundtruth", false, false, true);
Button save_traj_btn("ui.save_traj", &saveTrajectoryButton);

pangolin::Var<bool> follow("ui.follow", true, false, true);

// pangolin::Var<bool> record("ui.record", false, false, true);

pangolin::OpenGlRenderState camera;

// Visualization variables
std::unordered_map<int64_t, basalt::VioVisualizationData::Ptr> vis_map;

tbb::concurrent_bounded_queue<basalt::VioVisualizationData::Ptr> out_vis_queue;
tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr> out_state_queue;
tbb::concurrent_bounded_queue<MargData::Ptr> marg_queue;

std::vector<int64_t> vio_t_ns;
Eigen::aligned_vector<Eigen::Vector3d> vio_t_w_i;
Eigen::aligned_vector<Sophus::SE3d> vio_T_w_i;

std::vector<int64_t> gt_t_ns;
Eigen::aligned_vector<Eigen::Vector3d> gt_t_w_i;

std::string marg_data_path;
size_t last_frame_processed = 0;

tbb::concurrent_unordered_map<int64_t, int, std::hash<int64_t>> timestamp_to_id;

std::mutex m;
std::condition_variable cvar;
bool step_by_step = false;
size_t max_frames = 0;

std::atomic<bool> terminate = false;

// VIO variables
basalt::Calibration<double> calib;

basalt::VioDatasetPtr vio_dataset;
basalt::VioConfig vio_config;
basalt::OpticalFlowBase::Ptr opt_flow_ptr;
basalt::VioEstimatorBase::Ptr vio;
basalt::NfrMapper::Ptr nrf_mapper;

// Feed functions
void feed_images() {
  std::cout << "Started input_data thread " << std::endl;

  int NUM_CAMS = calib.intrinsics.size();
  for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
    if (vio->finished || terminate || (max_frames > 0 && i >= max_frames)) {
      // stop loop early if we set a limit on number of frames to process
      break;
    }

    if (step_by_step) {
      std::unique_lock<std::mutex> lk(m);
      cvar.wait(lk);
    }

    basalt::OpticalFlowInput::Ptr img(new basalt::OpticalFlowInput(NUM_CAMS));

    img->t_ns = vio_dataset->get_image_timestamps()[i];
    img->img_data = vio_dataset->get_image_data(img->t_ns);

    timestamp_to_id[img->t_ns] = i;

    opt_flow_ptr->input_img_queue.push(img);
  }

  // Indicate the end of the sequence
  opt_flow_ptr->input_img_queue.push(nullptr);

  std::cout << "Finished input_data thread " << std::endl;
}

void feed_imu() {
  for (size_t i = 0; i < vio_dataset->get_gyro_data().size(); i++) {
    if (vio->finished || terminate) {
      break;
    }

    basalt::ImuData<double>::Ptr data(new basalt::ImuData<double>);
    data->t_ns = vio_dataset->get_gyro_data()[i].timestamp_ns;

    data->accel = vio_dataset->get_accel_data()[i].data;
    data->gyro = vio_dataset->get_gyro_data()[i].data;

    vio->imu_data_queue.push(data);
    opt_flow_ptr->input_imu_queue.push(data);
  }
  vio->imu_data_queue.push(nullptr);
  opt_flow_ptr->input_imu_queue.push(nullptr);
}

int main(int argc, char** argv) {
  bool show_gui = true;
  bool show_map = false;
  bool print_queue = false;
  std::string cam_calib_path;
  std::string dataset_path;
  std::string dataset_type;
  std::string config_path;
  std::string result_path;
  std::string trajectory_fmt;
  bool trajectory_groundtruth;
  int num_threads = 0;
  bool use_imu = true;
  bool use_double = false;

  CLI::App app{"App description"};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--show-map", show_map, "Show Map");
  app.add_option("--cam-calib", cam_calib_path, "Ground-truth camera calibration used for simulation.")->required();

  app.add_option("--dataset-path", dataset_path, "Path to dataset.")->required();

  app.add_option("--dataset-type", dataset_type, "Dataset type <euroc, bag>.")->required();

  app.add_option("--marg-data", marg_data_path, "Path to folder where marginalization data will be stored.");

  app.add_option("--print-queue", print_queue, "Print queue.");
  app.add_option("--config-path", config_path, "Path to config file.");
  app.add_option("--result-path", result_path, "Path to result file where the system will write RMSE ATE.");
  app.add_option("--num-threads", num_threads, "Number of threads.");
  app.add_option("--step-by-step", step_by_step, "Path to config file.");
  app.add_option("--save-trajectory", trajectory_fmt, "Save trajectory. Supported formats <tum, euroc, kitti>");
  app.add_option("--save-groundtruth", trajectory_groundtruth, "In addition to trajectory, save also ground turth");
  app.add_option("--use-imu", use_imu, "Use IMU.");
  app.add_option("--use-double", use_double, "Use double not float.");
  app.add_option("--max-frames", max_frames, "Limit number of frames to process from dataset (0 means unlimited)");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  // global thread limit is in effect until global_control object is destroyed
  std::unique_ptr<tbb::global_control> tbb_global_control;
  if (num_threads > 0) {
    tbb_global_control =
        std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, num_threads);
  }

  if (!config_path.empty()) {
    vio_config.load(config_path);

    if (vio_config.vio_enforce_realtime) {
      vio_config.vio_enforce_realtime = false;
      std::cout << "The option vio_config.vio_enforce_realtime was enabled, "
                   "but it should only be used with the live executables (supply "
                   "images at a constant framerate). This executable runs on the "
                   "datasets and processes images as fast as it can, so the option "
                   "will be disabled. "
                << std::endl;
    }
  }

  load_data(cam_calib_path);

  {
    basalt::DatasetIoInterfacePtr dataset_io = basalt::DatasetIoFactory::getDatasetIo(dataset_type);

    dataset_io->read(dataset_path);

    vio_dataset = dataset_io->get_data();

    show_frame.Meta().range[1] = vio_dataset->get_image_timestamps().size() - 1;
    show_frame.Meta().gui_changed = true;

    opt_flow_ptr = basalt::OpticalFlowFactory::getOpticalFlow(vio_config, calib);
    opt_flow_ptr->start();

    for (size_t i = 0; i < vio_dataset->get_gt_pose_data().size(); i++) {
      gt_t_ns.push_back(vio_dataset->get_gt_timestamps()[i]);
      gt_t_w_i.push_back(vio_dataset->get_gt_pose_data()[i].translation());
    }
  }

  const int64_t start_t_ns = vio_dataset->get_image_timestamps().front();
  {
    vio = basalt::VioEstimatorFactory::getVioEstimator(vio_config, calib, basalt::constants::g, use_imu, use_double);
    vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    opt_flow_ptr->output_queue = &vio->vision_data_queue;
    opt_flow_ptr->show_gui = show_gui;
    if (show_gui) vio->out_vis_queue = &out_vis_queue;
    vio->out_state_queue = &out_state_queue;
    vio->opt_flow_depth_guess_queue = &opt_flow_ptr->input_depth_queue;
    vio->opt_flow_state_queue = &opt_flow_ptr->input_state_queue;
  }

  basalt::MargDataSaver::Ptr marg_data_saver;

  if (!marg_data_path.empty()) {
    marg_data_saver.reset(new basalt::MargDataSaver(marg_data_path));
    vio->out_marg_queue = &marg_data_saver->in_marg_queue;

    // Save gt.
    {
      std::string p = marg_data_path + "/gt.cereal";
      std::ofstream os(p, std::ios::binary);

      {
        cereal::BinaryOutputArchive archive(os);
        archive(gt_t_ns);
        archive(gt_t_w_i);
      }
      os.close();
    }
  }
  else if (show_map) {
    vio->out_marg_queue = &marg_queue;
  }

  vio_data_log.Clear();

  std::thread t1(&feed_images);
  std::thread t2(&feed_imu);
  std::shared_ptr<std::thread> map_thread;
  if (show_map) {
    map_thread.reset(new std::thread([&]() {
      initializeMapper();
    }));
  }

  std::shared_ptr<std::thread> t3;

  if (show_gui)
    t3.reset(new std::thread([&]() {
      basalt::VioVisualizationData::Ptr data;

      while (true) {
        out_vis_queue.pop(data);

        if (data.get()) {
          vis_map[data->t_ns] = data;
        } else {
          break;
        }
      }

      std::cout << "Finished t3" << std::endl;
    }));

  std::thread t4([&]() {
    basalt::PoseVelBiasState<double>::Ptr data;

    while (true) {
      out_state_queue.pop(data);

      if (!data.get()) break;

      int64_t t_ns = data->t_ns;

      // std::cerr << "t_ns " << t_ns << std::endl;
      Sophus::SE3d T_w_i = data->T_w_i;
      Eigen::Vector3d vel_w_i = data->vel_w_i;
      Eigen::Vector3d bg = data->bias_gyro;
      Eigen::Vector3d ba = data->bias_accel;

      vio_t_ns.emplace_back(data->t_ns);
      vio_t_w_i.emplace_back(T_w_i.translation());
      vio_T_w_i.emplace_back(T_w_i);

      if (show_gui) {
        std::vector<float> vals;
        vals.push_back((t_ns - start_t_ns) * 1e-9);

        for (int i = 0; i < 3; i++) vals.push_back(vel_w_i[i]);
        for (int i = 0; i < 3; i++) vals.push_back(T_w_i.translation()[i]);
        for (int i = 0; i < 3; i++) vals.push_back(bg[i]);
        for (int i = 0; i < 3; i++) vals.push_back(ba[i]);

        vio_data_log.Log(vals);
      }
    }

    std::cout << "Finished t4" << std::endl;
  });

  std::shared_ptr<std::thread> t5;

  auto print_queue_fn = [&]() {
    std::cout << "opt_flow_ptr->input_img_queue " << opt_flow_ptr->input_img_queue.size()
              << " opt_flow_ptr->output_queue " << opt_flow_ptr->output_queue->size() << " out_state_queue "
              << out_state_queue.size() << " imu_data_queue " << vio->imu_data_queue.size() << std::endl;
  };

  if (print_queue) {
    t5.reset(new std::thread([&]() {
      while (!terminate) {
        print_queue_fn();
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }));
  }

  auto time_start = std::chrono::high_resolution_clock::now();

  // record if we close the GUI before VIO is finished.
  bool aborted = false;

  if (show_gui) {
    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    pangolin::View& main_display = pangolin::CreateDisplay().SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    pangolin::View& img_view_display =
        pangolin::CreateDisplay().SetBounds(0.4, 1.0, 0.0, 0.4).SetLayout(pangolin::LayoutEqual);

    pangolin::View& plot_display = pangolin::CreateDisplay().SetBounds(0.0, 0.4, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    plotter = new pangolin::Plotter(&imu_data_log, 0.0, 100, -10.0, 10.0, 0.01f, 0.01f);
    plot_display.AddDisplay(*plotter);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < calib.intrinsics.size()) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);
      iv->UseNN() = true;  // Disable antialiasing, can be toggled with N key

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function = std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    Eigen::Vector3d cam_p(-0.5, -3, -5);
    cam_p = vio->getT_w_i_init().so3() * calib.T_i_c[0].so3() * cam_p;

    camera =
        pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
                                    pangolin::ModelViewLookAt(cam_p[0], cam_p[1], cam_p[2], 0, 0, 0, pangolin::AxisZ));

    pangolin::View& display3D = pangolin::CreateDisplay()
                                    .SetAspect(-640 / 480.0)
                                    .SetBounds(0.4, 1.0, 0.4, 1.0)
                                    .SetHandler(new pangolin::Handler3D(camera));

    display3D.extern_draw_function = draw_scene;

    main_display.AddDisplay(img_view_display);
    main_display.AddDisplay(display3D);

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      size_t frame_id = show_frame;
      int64_t t_ns = vio_dataset->get_image_timestamps()[frame_id];
      auto it = vis_map.find(t_ns);
      if (follow) {
        if (it != vis_map.end()) {
          Sophus::SE3d T_w_i;
          if (!it->second->states.empty()) {
            T_w_i = it->second->states.back();
          } else if (!it->second->frames.empty()) {
            T_w_i = it->second->frames.back();
          }
          T_w_i.so3() = Sophus::SO3d();

          camera.Follow(T_w_i.matrix());
        }
      }

      display3D.Activate(camera);
      glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

      img_view_display.Activate();
      if (fixed_depth.GuiChanged() && vio->opt_flow_depth_guess_queue != nullptr) {
        vio->opt_flow_depth_guess_queue->push(fixed_depth);
        depth_guess = fixed_depth;
      } else if (it != vis_map.end() && it->second->opt_flow_res && it->second->opt_flow_res->input_images) {
        depth_guess = it->second->opt_flow_res->input_images->depth_guess;
      }

      if (show_frame.GuiChanged()) {
        for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
          size_t frame_id = static_cast<size_t>(show_frame);
          int64_t timestamp = vio_dataset->get_image_timestamps()[frame_id];

          std::vector<basalt::ImageData> img_vec = vio_dataset->get_image_data(timestamp);

          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          fmt.gltype = GL_UNSIGNED_SHORT;
          fmt.scalable_internal_format = GL_LUMINANCE16;

          if (img_vec[cam_id].img.get())
            img_view[cam_id]->SetImage(img_vec[cam_id].img->ptr, img_vec[cam_id].img->w, img_vec[cam_id].img->h,
                                       img_vec[cam_id].img->pitch, fmt);
        }

        draw_plots();
      }

      if (show_est_vel.GuiChanged() || show_est_pos.GuiChanged() || show_est_ba.GuiChanged() ||
          show_est_bg.GuiChanged()) {
        draw_plots();
      }

      if (euroc_fmt.GuiChanged()) {
        euroc_fmt = true;
        tum_rgbd_fmt = false;
        kitti_fmt = false;
      }

      if (tum_rgbd_fmt.GuiChanged()) {
        tum_rgbd_fmt = true;
        euroc_fmt = false;
        kitti_fmt = false;
      }

      if (kitti_fmt.GuiChanged()) {
        kitti_fmt = true;
        euroc_fmt = false;
        tum_rgbd_fmt = false;
      }

      //      if (record) {
      //        main_display.RecordOnRender(
      //            "ffmpeg:[fps=50,bps=80000000,unique_filename]///tmp/"
      //            "vio_screencap.avi");
      //        record = false;
      //      }

      pangolin::FinishFrame();

      if (continue_btn) {
        if (!next_step()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }

      if (continue_fast) {
        int64_t t_ns = vio->last_processed_t_ns;
        if (timestamp_to_id.count(t_ns)) {
          show_frame = timestamp_to_id[t_ns];
          show_frame.Meta().gui_changed = true;
        }

        if (vio->finished) {
          continue_fast = false;
        }
      }
    }

    // If GUI closed but VIO not yet finished --> abort input queues, which in
    // turn aborts processing
    if (!vio->finished) {
      std::cout << "GUI closed but odometry still running --> aborting.\n";
      print_queue_fn();  // print queue size at time of aborting
      terminate = true;
      aborted = true;
    }
  }

  // wait first for vio to complete processing
  vio->maybe_join();

  // input threads will abort when vio is finished, but might be stuck in full
  // push to full queue, so drain queue now
  vio->drain_input_queues();

  // join input threads
  t1.join();
  t2.join();

  // std::cout << "Data input finished, terminate auxiliary threads.";
  terminate = true;

  // join other threads
  if (t3) t3->join();
  t4.join();
  if (t5) t5->join();
  if (map_thread) map_thread->join();

  // after joining all threads, print final queue sizes.
  if (print_queue) {
    std::cout << "Final queue sizes:" << std::endl;
    print_queue_fn();
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  const double duration_total = std::chrono::duration<double>(time_end - time_start).count();

  // TODO: remove this unconditional call (here for debugging);
  const double ate_rmse = basalt::alignSVD(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i);
  vio->debug_finalize();
  std::cout << "Total runtime: {:.3f}s\n"_format(duration_total);

  {
    basalt::ExecutionStats stats;
    stats.add("exec_time_s", duration_total);
    stats.add("ate_rmse", ate_rmse);
    stats.add("ate_num_kfs", vio_t_w_i.size());
    stats.add("num_frames", vio_dataset->get_image_timestamps().size());

    {
      basalt::MemoryInfo mi;
      if (get_memory_info(mi)) {
        stats.add("resident_memory_peak", mi.resident_memory_peak);
      }
    }

    stats.save_json("stats_vio.json");
  }

  if (!aborted && !trajectory_fmt.empty()) {
    std::cout << "Saving trajectory..." << std::endl;

    if (trajectory_fmt == "kitti") {
      kitti_fmt = true;
      euroc_fmt = false;
      tum_rgbd_fmt = false;
    }
    if (trajectory_fmt == "euroc") {
      euroc_fmt = true;
      kitti_fmt = false;
      tum_rgbd_fmt = false;
    }
    if (trajectory_fmt == "tum") {
      tum_rgbd_fmt = true;
      euroc_fmt = false;
      kitti_fmt = false;
    }

    save_groundtruth = trajectory_groundtruth;

    saveTrajectoryButton();
  }

  if (!aborted && !result_path.empty()) {
    double error = basalt::alignSVD(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i);

    auto exec_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start);

    std::ofstream os(result_path);
    {
      cereal::JSONOutputArchive ar(os);
      ar(cereal::make_nvp("rms_ate", error));
      ar(cereal::make_nvp("num_frames", vio_dataset->get_image_timestamps().size()));
      ar(cereal::make_nvp("exec_time_ns", exec_time_ns.count()));
    }
    os.close();
  }

  return 0;
}

void draw_image_overlay(pangolin::View& v, size_t cam_id) {
  UNUSED(v);

  //  size_t frame_id = show_frame;
  //  basalt::TimeCamId tcid =
  //      std::make_pair(vio_dataset->get_image_timestamps()[frame_id],
  //      cam_id);

  size_t frame_id = show_frame;
  auto it = vis_map.find(vio_dataset->get_image_timestamps()[frame_id]);
  if (it == vis_map.end()) return;
  basalt::VioVisualizationData::Ptr curr_vis_data = it->second;

  if (!curr_vis_data ||                                               //
      !curr_vis_data->opt_flow_res ||                                 //
      !curr_vis_data->opt_flow_res->input_images ||                   //
      curr_vis_data->opt_flow_res->input_images->img_data.empty() ||  //
      !curr_vis_data->opt_flow_res->input_images->img_data.at(0).img) {
    return;
  }
  const auto& frames = curr_vis_data->opt_flow_res->input_images->img_data;
  const auto& cam0_img = frames.at(0).img;

  size_t NUM_CAMS = curr_vis_data->projections->size();

  if (show_obs) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (cam_id < NUM_CAMS) {
      const auto& points = curr_vis_data->projections->at(cam_id);

      if (points.size() > 0) {
        double min_id = points[0][2], max_id = points[0][2];

        for (const auto& points2 : *curr_vis_data->projections)
          for (const auto& p : points2) {
            min_id = std::min(min_id, p[2]);
            max_id = std::max(max_id, p[2]);
          }

        for (const auto& c : points) {
          double u = c[0];
          double v = c[1];
          double depth = c[2] != 0.0 ? 1.0 / c[2] : std::numeric_limits<double>::infinity();
          int id = c[3];

          double width = cam0_img->w;
          double unit_radius = width / 96;
          double radius = unit_radius / depth;

          double min_depth = 1.0 / 3;  // 1/3 comes from how valid_kp is
                                       // computed in sqrt_keypoint_vio.cpp
          double max_depth = 20;       // And this is arbitrary
          double max_radius = unit_radius / min_depth;
          double min_radius = unit_radius * min_depth;

          bool clamped = depth < min_depth || depth > max_depth;
          double cradius = std::clamp(radius, min_radius, max_radius);

          float t = (cradius - min_radius) / (max_radius - min_radius);
          auto [r, g, b] = color_lerp(t);

          if (clamped) {  // Mark clamped points in UI
            glColor4f(r, g, b, 0.15);
            pangolin::glDrawCircle(u, v, cradius);
            glColor4f(r, g, b, 1);
          } else {
            glColor4f(r, g, b, 1);
            pangolin::glDrawCirclePerimeter(u, v, cradius);
          }

          if (show_ids) pangolin::GlFont::I().Text("%d", id).Draw(u, v);
          if (show_depth) pangolin::GlFont::I().Text("%.3lf m", depth).Draw(u, v + 5);
        }
      }

      if (show_guesses && cam_id != 0) {
        const auto keypoints0 = curr_vis_data->projections->at(0);
        const auto keypoints1 = curr_vis_data->projections->at(cam_id);

        double avg_invdepth = 0;
        double num_features = 0;
        for (const auto& cam_projs : *curr_vis_data->projections) {
          for (const Vector4d& v : cam_projs) avg_invdepth += v.z();
          num_features += cam_projs.size();
        }
        bool valid = avg_invdepth > 0 && num_features > 0;
        float default_depth = vio_config.optical_flow_matching_default_depth;
        double avg_depth = valid ? num_features / avg_invdepth : default_depth;

        for (const Vector4d& kp1 : keypoints1) {
          double u1 = kp1.x();
          double v1 = kp1.y();
          // double invdist1 = kp1.z();
          double id1 = kp1.w();

          double u0 = 0;
          double v0 = 0;
          bool found = false;
          for (const Vector4d& kp0 : keypoints0) {  // Find match in keypoints0
            double id0 = kp0.w();
            if (id1 != id0) continue;
            u0 = kp0.x();
            v0 = kp0.y();
            found = true;
            break;
          }

          // Display guess error if this is a stereo feature
          // NOTE: keep in mind that these guesses are not really the guesses
          // used to detect the feature, but the guess we would use if we were
          // to detect the feature right now.
          if (found) {
            // Guess if we were using SAME_PIXEL
            if (show_same_pixel_guess) {
              glColor3f(0, 1, 1);  // Cyan
              pangolin::glDrawLine(u1, v1, u0, v0);
            }

            // Guess if we were using REPROJ_FIX_DEPTH
            if (show_reproj_fix_depth_guess) {
              glColor3f(1, 1, 0);  // Yellow
              auto off = calib.viewOffset({u0, v0}, fixed_depth, 0, cam_id);
              pangolin::glDrawLine(u1, v1, u0 - off.x(), v0 - off.y());
            }

            // Guess if we were using REPROJ_AVG_DEPTH
            if (show_reproj_avg_depth_guess) {
              glColor3f(1, 0, 1);  // Magenta
              auto off = calib.viewOffset({u0, v0}, avg_depth, 0, cam_id);
              pangolin::glDrawLine(u1, v1, u0 - off.x(), v0 - off.y());
            }

            // Guess with the current guess type
            if (show_active_guess) {
              glColor3f(1, 0, 0);  // Red
              Vector2d off{0, 0};
              if (vio_config.optical_flow_matching_guess_type != MatchingGuessType::SAME_PIXEL) {
                off = calib.viewOffset({u0, v0}, curr_vis_data->opt_flow_res->input_images->depth_guess, 0, cam_id);
              }
              pangolin::glDrawLine(u1, v1, u0 - off.x(), v0 - off.y());
            }
          }
        }
      }
      glColor3f(1.0, 0.0, 0.0);
      pangolin::GlFont::I().Text("Tracked %d points", points.size()).Draw(5, 20);
    }
  }

  if (show_flow) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const Keypoints& kp_map = curr_vis_data->opt_flow_res->keypoints[cam_id];

    for (const auto& kv : kp_map) {
      Eigen::MatrixXf transformed_patch = kv.second.linear() * opt_flow_ptr->patch_coord;
      transformed_patch.colwise() += kv.second.translation();

      for (int i = 0; i < transformed_patch.cols(); i++) {
        const Eigen::Vector2f c = transformed_patch.col(i);
        pangolin::glDrawCirclePerimeter(c[0], c[1], 0.5f);
      }

      const Eigen::Vector2f c = kv.second.translation();

      if (show_ids) pangolin::GlFont::I().Text("%d", kv.first).Draw(5 + c[0], 5 + c[1]);
    }

    pangolin::GlFont::I().Text("%d opt_flow patches", kp_map.size()).Draw(5, 20);
  }

  if (show_tracking_guess) {
    size_t frame_id = show_frame;
    if (frame_id < 1) goto out_show_tracking_guess;

    int64_t now_ts = vio_dataset->get_image_timestamps().at(frame_id);
    int64_t prev_ts = vio_dataset->get_image_timestamps().at(frame_id - 1);

    auto now_it = vis_map.find(now_ts);
    auto prev_it = vis_map.find(prev_ts);

    auto end_it = vis_map.end();
    if (now_it == end_it || prev_it == end_it) goto out_show_tracking_guess;

    auto new_kpts = now_it->second->opt_flow_res->keypoints[cam_id];
    auto prev_kpts = prev_it->second->opt_flow_res->keypoints[cam_id];
    auto guess_obs = now_it->second->opt_flow_res->tracking_guesses[cam_id];

    std::vector<Vector2f> prev_lines;
    std::vector<Vector2f> prev_points;
    std::vector<Vector2f> guess_lines;
    std::vector<Vector2f> guess_points;
    std::vector<Vector2f> now_points;

    prev_lines.reserve(new_kpts.size());
    prev_points.reserve(new_kpts.size());
    guess_lines.reserve(new_kpts.size());
    guess_points.reserve(new_kpts.size());
    now_points.reserve(new_kpts.size());

    float radius = 3.0f;

    // Draw tracked features in previous frame
    for (auto& [kpid, kpt] : new_kpts) {
      if (prev_kpts.count(kpid) == 0) continue;
      if (guess_obs.count(kpid) == 0) continue;

      auto n = kpt.translation();
      auto p = prev_kpts.at(kpid).translation();
      auto g = guess_obs.at(kpid).translation();

      now_points.emplace_back(n);

      prev_lines.emplace_back(p);
      prev_lines.emplace_back(n);
      prev_points.emplace_back(p);

      guess_lines.emplace_back(g);
      guess_lines.emplace_back(n);
      guess_points.emplace_back(g);
    }

    glColor4f(1, 0.59, 0, 0.9);
    glDrawCirclePerimeters(now_points, radius);

    glColor4f(0.93, 0.42, 0, 0.3);
    pangolin::glDrawLines(prev_lines);
    glDrawCirclePerimeters(prev_points, radius);

    glColor4f(1, 0.59, 0, 0.5);
    pangolin::glDrawLines(guess_lines);
    glDrawCirclePerimeters(guess_points, radius);
  }

out_show_tracking_guess:

  if (show_matching_guess) {
    auto new_kpts = curr_vis_data->opt_flow_res->keypoints[cam_id];
    auto cam0_kpts = curr_vis_data->opt_flow_res->keypoints[0];
    auto guess_obs = curr_vis_data->opt_flow_res->matching_guesses[cam_id];

    std::vector<Vector2f> cam0_lines;
    std::vector<Vector2f> cam0_points;
    std::vector<Vector2f> guess_lines;
    std::vector<Vector2f> guess_points;
    std::vector<Vector2f> now_points;

    cam0_lines.reserve(new_kpts.size());
    cam0_points.reserve(new_kpts.size());
    guess_lines.reserve(new_kpts.size());
    guess_points.reserve(new_kpts.size());
    now_points.reserve(new_kpts.size());

    float radius = 3.0f;

    // Draw tracked features in previous frame
    for (auto& [kpid, kpt] : new_kpts) {
      if (cam0_kpts.count(kpid) == 0) continue;
      if (guess_obs.count(kpid) == 0) continue;

      auto n = kpt.translation();
      auto c = cam0_kpts.at(kpid).translation();
      auto g = guess_obs.at(kpid).translation();

      now_points.emplace_back(n);

      cam0_lines.emplace_back(c);
      cam0_lines.emplace_back(n);
      cam0_points.emplace_back(c);

      guess_lines.emplace_back(g);
      guess_lines.emplace_back(n);
      guess_points.emplace_back(g);
    }

    glColor4f(0.12, 0.58, 0.95, 0.9);
    glDrawCirclePerimeters(now_points, radius);

    glColor4f(0, 0.73, 0.83, 0.5);
    pangolin::glDrawLines(cam0_lines);
    glDrawCirclePerimeters(cam0_points, radius);

    glColor4f(0.12, 0.58, 0.95, 0.5);
    pangolin::glDrawLines(guess_lines);
    glDrawCirclePerimeters(guess_points, radius);
  }

  if (show_masks) {
    glColor4f(0.0, 1.0, 1.0, 0.1);
    for (const Rect& m : curr_vis_data->opt_flow_res->input_images->masks[cam_id].masks) {
      pangolin::glDrawRect(m.x, m.y, m.x + m.w, m.y + m.h);
    }
  }

  int C = vio_config.optical_flow_detection_grid_size;

  int w = cam0_img->w;
  int h = cam0_img->h;

  int x_start = (w % C) / 2;
  int y_start = (h % C) / 2;

  int x_stop = x_start + C * (w / C - 1);
  int y_stop = y_start + C * (h / C - 1);

  int x_first = x_start + C / 2;
  int y_first = y_start + C / 2;

  int x_end = x_stop + C;
  int y_end = y_stop + C;

  int x_last = x_stop + C / 2;
  int y_last = y_stop + C / 2;

  if (show_cam0_proj) {
    std::vector<Vector2d> points;
    auto drawPoint = [&points, w, h, &curr_vis_data](float u, float v, int j, bool draw_c0_uv) {
      Vector2d ci_uv{u, v};
      Vector2d c0_uv;
      double _;
      bool projected = calib.projectBetweenCams(ci_uv, depth_guess, c0_uv, _, j, 0);
      bool in_bounds = c0_uv.x() >= 0 && c0_uv.x() < w && c0_uv.y() >= 0 && c0_uv.y() < h;
      bool valid = projected && in_bounds;

      // Define color
      GLfloat invalid_color[4] = {1, 0, 0, 0.5};      // red
      GLfloat in_bounds_color[4] = {1, 0.5, 0, 0.5};  // orange
      GLfloat projected_color[4] = {1, 0.9, 0, 0.5};  // yellow
      GLfloat valid_color[4] = {0, 1, 0, 0.5};        // green
      GLfloat* color = invalid_color;
      if (valid) {
        color = valid_color;
      } else if (projected) {
        color = projected_color;
      } else if (in_bounds) {
        color = in_bounds_color;
      }
      glColor4fv(color);

      // Press L key twice in viewer to be able to see out-of-bounds points
      if (projected) {
        points.push_back(c0_uv);
      }

      if (draw_c0_uv) {
        pangolin::glDrawCircle(c0_uv.x(), c0_uv.y(), 2);
      } else {
        pangolin::glDrawCircle(ci_uv.x(), ci_uv.y(), 2);
      }
    };

    if (cam_id == 0) {
      for (size_t target_cam = 1; target_cam < NUM_CAMS; target_cam++) {
        points.clear();

#if 1  // Draw perimeter of projected-to-cam0 grid
        int x = x_first;
        int y = y_first;
        for (; x <= x_last; x += C) drawPoint(x, y, target_cam, true);
        for (x = x_last; y <= y_last; y += C) drawPoint(x, y, target_cam, true);
        for (y = y_last; x >= x_first; x -= C) drawPoint(x, y, target_cam, true);
        for (x = x_first; y >= y_first; y -= C) drawPoint(x, y, target_cam, true);

#else  // Draw full projected-to-cam0 grid
        for (int y = x_first; y <= y_last; y += C) {
          for (int x = y_first; x <= x_last; x += C) {
            drawPoint(x, y, target_cam, true);
          }
        }
#endif

        glColor4f(0.0, 1.0, 0.0, 0.5);
        pangolin::glDrawLineLoop(points);
      }
    } else {
      for (int y = y_first; y < h; y += C) {
        for (int x = x_first; x < w; x += C) {
          drawPoint(x, y, cam_id, false);
        }
      }
    }
  }

  if (show_grid) {
    glColor4f(1.0, 0.0, 1.0, 0.25);

    std::vector<Vector2f> grid_lines;
    for (int x = x_start; x <= x_end; x += C) {
      grid_lines.emplace_back(x, y_start);
      grid_lines.emplace_back(x, y_end);
    }
    for (int y = y_start; y <= y_end; y += C) {
      grid_lines.emplace_back(x_start, y);
      grid_lines.emplace_back(x_end, y);
    }
    pangolin::glDrawLines(grid_lines);
  }
}

void draw_scene(pangolin::View& view) {
  UNUSED(view);
  view.Activate(camera);
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

  glPointSize(3);
  glColor3f(1.0, 0.0, 0.0);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glColor3ubv(cam_color);
  if (!vio_t_w_i.empty()) {
    size_t end = std::min(vio_t_w_i.size(), size_t(show_frame + 1));
    Eigen::aligned_vector<Eigen::Vector3d> sub_gt(vio_t_w_i.begin(), vio_t_w_i.begin() + end);
    pangolin::glDrawLineStrip(sub_gt);
  }

  glColor3ubv(gt_color);
  if (show_gt) pangolin::glDrawLineStrip(gt_t_w_i);

  size_t frame_id = show_frame;
  int64_t t_ns = vio_dataset->get_image_timestamps()[frame_id];
  auto it = vis_map.find(t_ns);

  if (it != vis_map.end()) {
    for (size_t i = 0; i < calib.T_i_c.size(); i++)
      if (!it->second->states.empty()) {
        render_camera((it->second->states.back() * calib.T_i_c[i]).matrix(), 2.0f, cam_color, 0.1f);
      } else if (!it->second->frames.empty()) {
        render_camera((it->second->frames.back() * calib.T_i_c[i]).matrix(), 2.0f, cam_color, 0.1f);
      }

    for (const auto& p : it->second->states)
      for (size_t i = 0; i < calib.T_i_c.size(); i++)
        render_camera((p * calib.T_i_c[i]).matrix(), 2.0f, state_color, 0.1f);

    for (const auto& p : it->second->frames)
      for (size_t i = 0; i < calib.T_i_c.size(); i++)
        render_camera((p * calib.T_i_c[i]).matrix(), 2.0f, pose_color, 0.1f);

    glColor3ubv(pose_color);
    pangolin::glDrawPoints(it->second->points);
  }

  pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);
}

void load_data(const std::string& calib_path) {
  std::ifstream os(calib_path, std::ios::binary);

  if (os.is_open()) {
    cereal::JSONInputArchive archive(os);
    archive(calib);
    std::cout << "Loaded camera with " << calib.intrinsics.size() << " cameras" << std::endl;

  } else {
    std::cerr << "could not load camera calibration " << calib_path << std::endl;
    std::abort();
  }
}

bool next_step() {
  if (show_frame < int(vio_dataset->get_image_timestamps().size()) - 1) {
    show_frame = show_frame + 1;
    show_frame.Meta().gui_changed = true;
    cvar.notify_one();
    return true;
  } else {
    return false;
  }
}

bool prev_step() {
  if (show_frame > 1) {
    show_frame = show_frame - 1;
    show_frame.Meta().gui_changed = true;
    return true;
  } else {
    return false;
  }
}

void draw_plots() {
  plotter->ClearSeries();
  plotter->ClearMarkers();

  if (show_est_pos) {
    plotter->AddSeries("$0", "$4", pangolin::DrawingModeLine, pangolin::Colour::Red(), "position x", &vio_data_log);
    plotter->AddSeries("$0", "$5", pangolin::DrawingModeLine, pangolin::Colour::Green(), "position y", &vio_data_log);
    plotter->AddSeries("$0", "$6", pangolin::DrawingModeLine, pangolin::Colour::Blue(), "position z", &vio_data_log);
  }

  if (show_est_vel) {
    plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine, pangolin::Colour::Red(), "velocity x", &vio_data_log);
    plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine, pangolin::Colour::Green(), "velocity y", &vio_data_log);
    plotter->AddSeries("$0", "$3", pangolin::DrawingModeLine, pangolin::Colour::Blue(), "velocity z", &vio_data_log);
  }

  if (show_est_bg) {
    plotter->AddSeries("$0", "$7", pangolin::DrawingModeLine, pangolin::Colour::Red(), "gyro bias x", &vio_data_log);
    plotter->AddSeries("$0", "$8", pangolin::DrawingModeLine, pangolin::Colour::Green(), "gyro bias y", &vio_data_log);
    plotter->AddSeries("$0", "$9", pangolin::DrawingModeLine, pangolin::Colour::Blue(), "gyro bias z", &vio_data_log);
  }

  if (show_est_ba) {
    plotter->AddSeries("$0", "$10", pangolin::DrawingModeLine, pangolin::Colour::Red(), "accel bias x", &vio_data_log);
    plotter->AddSeries("$0", "$11", pangolin::DrawingModeLine, pangolin::Colour::Green(), "accel bias y",
                       &vio_data_log);
    plotter->AddSeries("$0", "$12", pangolin::DrawingModeLine, pangolin::Colour::Blue(), "accel bias z", &vio_data_log);
  }

  double t = vio_dataset->get_image_timestamps()[show_frame] * 1e-9;
  plotter->AddMarker(pangolin::Marker::Vertical, t, pangolin::Marker::Equal, pangolin::Colour::White());
}

void alignButton() { basalt::alignSVD(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i); }

void saveTrajectoryButton() {
  if (tum_rgbd_fmt) {
    {
      std::ofstream os("trajectory.txt");

      os << "# timestamp tx ty tz qx qy qz qw" << std::endl;

      for (size_t i = 0; i < vio_t_ns.size(); i++) {
        const Sophus::SE3d& pose = vio_T_w_i[i];
        os << std::scientific << std::setprecision(18) << vio_t_ns[i] * 1e-9 << " " << pose.translation().x() << " "
           << pose.translation().y() << " " << pose.translation().z() << " " << pose.unit_quaternion().x() << " "
           << pose.unit_quaternion().y() << " " << pose.unit_quaternion().z() << " " << pose.unit_quaternion().w()
           << std::endl;
      }

      os.close();
    }

    if (save_groundtruth) {
      std::ofstream os("groundtruth.txt");

      os << "# timestamp tx ty tz qx qy qz qw" << std::endl;

      for (size_t i = 0; i < gt_t_ns.size(); i++) {
        const Eigen::Vector3d& pos = gt_t_w_i[i];
        os << std::scientific << std::setprecision(18) << gt_t_ns[i] * 1e-9 << " " << pos.x() << " " << pos.y() << " "
           << pos.z() << " "
           << "0 0 0 1" << std::endl;
      }

      os.close();
    }

    std::cout << "Saved trajectory in TUM RGB-D Dataset format in trajectory.txt" << std::endl;
  } else if (euroc_fmt) {
    std::ofstream os("trajectory.csv");

    os << "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w "
          "[],q_RS_x [],q_RS_y [],q_RS_z []"
       << std::endl;

    for (size_t i = 0; i < vio_t_ns.size(); i++) {
      const Sophus::SE3d& pose = vio_T_w_i[i];
      os << std::scientific << std::setprecision(18) << vio_t_ns[i] << "," << pose.translation().x() << ","
         << pose.translation().y() << "," << pose.translation().z() << "," << pose.unit_quaternion().w() << ","
         << pose.unit_quaternion().x() << "," << pose.unit_quaternion().y() << "," << pose.unit_quaternion().z()
         << std::endl;
    }

    std::cout << "Saved trajectory in Euroc Dataset format in trajectory.csv" << std::endl;
  } else {
    std::ofstream os("trajectory_kitti.txt");

    for (size_t i = 0; i < vio_t_ns.size(); i++) {
      Eigen::Matrix<double, 3, 4> mat = vio_T_w_i[i].matrix3x4();
      os << std::scientific << std::setprecision(12) << mat.row(0) << " " << mat.row(1) << " " << mat.row(2) << " "
         << std::endl;
    }

    os.close();

    std::cout << "Saved trajectory in KITTI Dataset format in trajectory_kitti.txt" << std::endl;
  }
}

/// --------------------   MAPPER GUI   --------------------

// Mapper GUI variables
pangolin::Var<int> map_show_frame1("map_ui.show_frame1", 0, 0, 1);
pangolin::Var<int> map_show_cam1("map_ui.show_cam1", 0, 0, 0);
pangolin::Var<int> map_show_frame2("map_ui.show_frame2", 0, 0, 1);
pangolin::Var<int> map_show_cam2("map_ui.show_cam2", 0, 0, 0);
pangolin::Var<bool> map_lock_frames("map_ui.lock_frames", true, false, true);
pangolin::Var<bool> map_show_detected("map_ui.show_detected", true, false, true);
pangolin::Var<bool> map_show_matches("map_ui.show_matches", true, false, true);
pangolin::Var<bool> map_show_inliers("map_ui.show_inliers", true, false, true);
pangolin::Var<bool> map_show_ids("map_ui.show_ids", false, false, true);

pangolin::Var<int> map_num_opt_iter("map_ui.num_opt_iter", 10, 0, 20);

pangolin::Var<bool> map_show_gt("map_ui.show_gt", true, false, true);
pangolin::Var<bool> map_show_edges("map_ui.show_edges", true, false, true);
pangolin::Var<bool> map_show_points("map_ui.show_points", true, false, true);

pangolin::OpenGlRenderState map_camera;

/// Mapper variables
Eigen::aligned_vector<Eigen::Vector3d> gt_frame_t_w_i;
std::vector<int64_t> gt_frame_t_ns, image_t_ns;

Eigen::aligned_vector<Eigen::Vector3d> mapper_points;
std::vector<int> mapper_point_ids;

std::map<int64_t, basalt::MargData::Ptr> marg_data;

Eigen::aligned_vector<Eigen::Vector3d> edges_vis;
Eigen::aligned_vector<Eigen::Vector3d> roll_pitch_vis;
Eigen::aligned_vector<Eigen::Vector3d> rel_edges_vis;


// Mapper functions

void initializeMapper() {
  nrf_mapper.reset(new basalt::NfrMapper(calib, vio_config));
  while (true) {
    basalt::MargData::Ptr data;
    marg_queue.pop(data);
    if (data.get() != nullptr) {
      nrf_mapper->addMargData(data);
      computeEdgeVis();
      {
        std::cout << "Loaded " << nrf_mapper->img_data.size() << " images." << std::endl;

        map_show_frame1.Meta().range[1] = nrf_mapper->img_data.size() - 1;
        map_show_frame2.Meta().range[1] = nrf_mapper->img_data.size() - 1;
        map_show_frame1.Meta().gui_changed = true;
        map_show_frame2.Meta().gui_changed = true;

        map_show_cam1.Meta().range[1] = calib.intrinsics.size() - 1;
        map_show_cam2.Meta().range[1] = calib.intrinsics.size() - 1;
        if (calib.intrinsics.size() > 1) map_show_cam2 = 1;

        for (const auto& kv : nrf_mapper->img_data) {
          image_t_ns.emplace_back(kv.first);
        }

        std::sort(image_t_ns.begin(), image_t_ns.end());
      }
    }
    else { break; }
  }
  showMap();
}

void loadMapData() {
  std::cout << "Loading marg data into the mapper..." << std::endl;
  while (true) {
    basalt::MargData::Ptr data;
    marg_queue.pop(data);
    if (data.get() != nullptr) {
      nrf_mapper->addMargData(data);
    }
    else { break; }
  }
}

void computeEdgeVis() {
  edges_vis.clear();
  // std::cout << "Landmark: " << nrf_mapper->lmdb.numLandmarks() << std::endl;
  for (const auto& kv1 : nrf_mapper->lmdb.getObservations()) {
    for (const auto& kv2 : kv1.second) {
      Eigen::Vector3d p1 = nrf_mapper->getFramePoses().at(kv1.first.frame_id).getPose().translation();
      Eigen::Vector3d p2 = nrf_mapper->getFramePoses().at(kv2.first.frame_id).getPose().translation();

      edges_vis.emplace_back(p1);
      edges_vis.emplace_back(p2);
    }
  }

  roll_pitch_vis.clear();
  for (const auto& v : nrf_mapper->roll_pitch_factors) {
    const Sophus::SE3d& T_w_i = nrf_mapper->getFramePoses().at(v.t_ns).getPose();

    Eigen::Vector3d p = T_w_i.translation();
    Eigen::Vector3d d = v.R_w_i_meas * T_w_i.so3().inverse() * (-Eigen::Vector3d::UnitZ());

    roll_pitch_vis.emplace_back(p);
    roll_pitch_vis.emplace_back(p + 0.1 * d);
  }

  rel_edges_vis.clear();
  for (const auto& v : nrf_mapper->rel_pose_factors) {
    Eigen::Vector3d p1 = nrf_mapper->getFramePoses().at(v.t_i_ns).getPose().translation();
    Eigen::Vector3d p2 = nrf_mapper->getFramePoses().at(v.t_j_ns).getPose().translation();

    rel_edges_vis.emplace_back(p1);
    rel_edges_vis.emplace_back(p2);
  }
}

void showMap() {
    std::cout << "Show map..." << std::endl;
    pangolin::CreateWindowAndBind("Map", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    pangolin::View& img_view_display = pangolin::CreateDisplay()
                                           .SetBounds(0.4, 1.0, pangolin::Attach::Pix(UI_WIDTH), 0.4)
                                           .SetLayout(pangolin::LayoutEqual);

    pangolin::CreatePanel("map_ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < calib.intrinsics.size()) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function = std::bind(&draw_mapper_image_overlay, std::placeholders::_1, idx);
    }

    map_camera =
        pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
                                    pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2, pangolin::AxisNegY));

    //    pangolin::OpenGlRenderState camera(
    //        pangolin::ProjectionMatrixOrthographic(-30, 30, -30, 30, -30, 30),
    //        pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,
    //                                  pangolin::AxisNegY));

    pangolin::View& display3D = pangolin::CreateDisplay()
                                    .SetAspect(-640 / 480.0)
                                    .SetBounds(0.0, 1.0, 0.4, 1.0)
                                    .SetHandler(new pangolin::Handler3D(map_camera));

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (map_lock_frames) {
        // in case of locking frames, chaning one should change the other
        if (map_show_frame1.GuiChanged()) {
          map_show_frame2 = map_show_frame1;
          map_show_frame2.Meta().gui_changed = true;
          map_show_frame1.Meta().gui_changed = true;
        } else if (map_show_frame2.GuiChanged()) {
          map_show_frame1 = map_show_frame2;
          map_show_frame1.Meta().gui_changed = true;
          map_show_frame2.Meta().gui_changed = true;
        }
      }

      display3D.Activate(map_camera);
      glClearColor(1.f, 1.f, 1.f, 1.0f);

      draw_map_scene();

      if (map_show_frame1.GuiChanged() || map_show_cam1.GuiChanged()) {
        size_t frame_id = static_cast<size_t>(map_show_frame1);
        int64_t timestamp = image_t_ns[frame_id];
        size_t cam_id = map_show_cam1;

        if (nrf_mapper->img_data.count(timestamp) > 0 && nrf_mapper->img_data.at(timestamp).get()) {
          const std::vector<basalt::ImageData>& img_vec = nrf_mapper->img_data.at(timestamp)->img_data;

          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          fmt.gltype = GL_UNSIGNED_SHORT;
          fmt.scalable_internal_format = GL_LUMINANCE16;

          if (img_vec[cam_id].img.get()) {
            img_view[0]->SetImage(img_vec[cam_id].img->ptr, img_vec[cam_id].img->w, img_vec[cam_id].img->h,
                                  img_vec[cam_id].img->pitch, fmt);
          } else {
            img_view[0]->Clear();
          }
        } else {
          img_view[0]->Clear();
        }
      }

      if (euroc_fmt.GuiChanged()) {
        tum_rgbd_fmt = !euroc_fmt;
      }

      if (tum_rgbd_fmt.GuiChanged()) {
        euroc_fmt = !tum_rgbd_fmt;
      }

      if (map_show_frame2.GuiChanged() || map_show_cam2.GuiChanged()) {
        size_t frame_id = static_cast<size_t>(map_show_frame2);
        int64_t timestamp = image_t_ns[frame_id];
        size_t cam_id = map_show_cam2;

        if (nrf_mapper->img_data.count(timestamp) > 0 && nrf_mapper->img_data.at(timestamp).get()) {
          const std::vector<basalt::ImageData>& img_vec = nrf_mapper->img_data.at(timestamp)->img_data;

          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          fmt.gltype = GL_UNSIGNED_SHORT;
          fmt.scalable_internal_format = GL_LUMINANCE16;

          if (img_vec[cam_id].img.get()) {
            img_view[1]->SetImage(img_vec[cam_id].img->ptr, img_vec[cam_id].img->w, img_vec[cam_id].img->h,
                                  img_vec[cam_id].img->pitch, fmt);
          } else {
            img_view[1]->Clear();
          }
        } else {
          img_view[1]->Clear();
        }
      }

      pangolin::FinishFrame();

      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void draw_mapper_image_overlay(pangolin::View& v, size_t view_id) {
  UNUSED(v);

  size_t frame_id = (view_id == 0) ? map_show_frame1 : map_show_frame2;
  size_t cam_id = (view_id == 0) ? map_show_cam1 : map_show_cam2;

  basalt::TimeCamId tcid(image_t_ns[frame_id], cam_id);

  float text_row = 20;

  if (map_show_detected) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);  // red
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (nrf_mapper->feature_corners.find(tcid) != nrf_mapper->feature_corners.end()) {
      const basalt::KeypointsData& cr = nrf_mapper->feature_corners.at(tcid);

      for (size_t i = 0; i < cr.corners.size(); i++) {
        Eigen::Vector2d c = cr.corners[i];
        double angle = cr.corner_angles[i];
        pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

        Eigen::Vector2d r(3, 0);
        Eigen::Rotation2Dd rot(angle);
        r = rot * r;

        pangolin::glDrawLine(c, c + r);
      }

      pangolin::GlFont::I().Text("Detected %d corners", cr.corners.size()).Draw(5, 20);

    } else {
      glLineWidth(1.0);

      pangolin::GlFont::I().Text("Corners not processed").Draw(5, text_row);
    }
    text_row += 20;
  }

  if (map_show_matches || map_show_inliers) {
    glLineWidth(1.0);
    glColor3f(0.0, 0.0, 1.0);  // blue
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    size_t o_frame_id = (view_id == 0 ? map_show_frame2 : map_show_frame1);
    size_t o_cam_id = (view_id == 0 ? map_show_cam2 : map_show_cam1);

    basalt::TimeCamId o_tcid(image_t_ns[o_frame_id], o_cam_id);

    int idx = -1;

    auto it = nrf_mapper->feature_matches.find(std::make_pair(tcid, o_tcid));

    if (it != nrf_mapper->feature_matches.end()) {
      idx = 0;
    } else {
      it = nrf_mapper->feature_matches.find(std::make_pair(o_tcid, tcid));
      if (it != nrf_mapper->feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && map_show_matches) {
      if (nrf_mapper->feature_corners.find(tcid) != nrf_mapper->feature_corners.end()) {
        const basalt::KeypointsData& cr = nrf_mapper->feature_corners.at(tcid);

        for (size_t i = 0; i < it->second.matches.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.matches[i].first : it->second.matches[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (map_show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I().Text("Detected %d matches", it->second.matches.size()).Draw(5, text_row);
        text_row += 20;
      }
    }

    glColor3f(0.0, 1.0, 0.0);  // green

    if (idx >= 0 && map_show_inliers) {
      if (nrf_mapper->feature_corners.find(tcid) != nrf_mapper->feature_corners.end()) {
        const basalt::KeypointsData& cr = nrf_mapper->feature_corners.at(tcid);

        for (size_t i = 0; i < it->second.inliers.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.inliers[i].first : it->second.inliers[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I().Text("Detected %d inliers", it->second.inliers.size()).Draw(5, text_row);
        text_row += 20;
      }
    }
  }
}

void draw_map_scene() {
  glPointSize(3);
  glColor3f(1.0, 0.0, 0.0);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glColor3ubv(pose_color);
  if (map_show_points) pangolin::glDrawPoints(mapper_points);

  glColor3ubv(gt_color);
  if (show_gt) pangolin::glDrawLineStrip(gt_frame_t_w_i);

  glColor3f(0.0, 1.0, 0.0);
  if (map_show_edges) pangolin::glDrawLines(edges_vis);

  glLineWidth(2);
  glColor3f(1.0, 0.0, 1.0);
  if (map_show_edges) pangolin::glDrawLines(roll_pitch_vis);
  glLineWidth(1);

  glColor3f(1.0, 0.0, 0.0);
  if (map_show_edges) pangolin::glDrawLines(rel_edges_vis);

  for (const auto& kv : nrf_mapper->getFramePoses()) {
    pangolin::glDrawAxis(kv.second.getPose().matrix(), 0.1);
  }

  pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);
}
