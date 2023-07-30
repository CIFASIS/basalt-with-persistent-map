#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#include <CLI/CLI.hpp>

#include <sophus/se3.hpp>

#include <pangolin/display/display.h>
#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/pangolin.h>

#include <basalt/io/marg_data_io.h>
#include <basalt/optical_flow/optical_flow.h>
#include <basalt/serialization/headers_serialization.h>
#include <basalt/utils/keypoints.h>
#include <basalt/utils/vio_config.h>
#include <basalt/utils/vis_utils.h>
#include <basalt/vi_estimator/vio_estimator.h>

#define ASSERT(cond, ...)                                      \
  do {                                                         \
    if (!(cond)) {                                             \
      printf("Assertion failed @%s:%d\n", __func__, __LINE__); \
      printf(__VA_ARGS__);                                     \
      printf("\n");                                            \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  } while (false);
#define ASSERT_(cond) ASSERT(cond, "%s", #cond);

namespace xrt::auxiliary::tracking::slam {

using std::cout;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::thread;
using std::to_string;
using std::vector;
using namespace basalt;
using namespace Eigen;

class slam_tracker_ui {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  VioVisualizationData::Ptr curr_vis_data = nullptr;
  VioVisualizationData::Ptr prev_vis_data = nullptr;
  pangolin::DataLog vio_data_log;
  thread vis_thread;
  thread ui_runner_thread;
  size_t num_cams = 0;
  std::atomic<bool> running = false;

 public:
  tbb::concurrent_bounded_queue<VioVisualizationData::Ptr> out_vis_queue{};
  tbb::concurrent_queue<double> *opt_flow_depth_queue = nullptr;
  OpticalFlowBase::Ptr opt_flow;

  void initialize(int ncams) {
    vio_data_log.Clear();
    ASSERT_(ncams > 0);
    num_cams = ncams;
  }

  void start(const Sophus::SE3d &T_w_i_init, const Calibration<double> &calib, const VioConfig &config,
             tbb::concurrent_queue<double> *ofq_depth, OpticalFlowBase::Ptr of) {
    opt_flow_depth_queue = ofq_depth;
    opt_flow = of;
    running = true;
    start_visualization_thread();
    start_ui(T_w_i_init, calib, config);
  }

  void stop() {
    running = false;
    vis_thread.join();
    ui_runner_thread.join();
  }

  void start_visualization_thread() {
    vis_thread = thread([&]() {
      while (true) {
        auto curr = curr_vis_data;
        out_vis_queue.pop(curr_vis_data);
        show_frame = show_frame + 1;
        if (curr_vis_data.get() == nullptr) break;
        prev_vis_data = curr;
      }
      cout << "Finished vis_thread\n";
    });
  }

  int64_t start_t_ns = -1;
  std::vector<int64_t> vio_t_ns;
  Eigen::aligned_vector<Eigen::Vector3d> vio_t_w_i;

  void log_vio_data(const PoseVelBiasState<double>::Ptr &data) {
    int64_t t_ns = data->t_ns;
    if (start_t_ns < 0) start_t_ns = t_ns;
    float since_start_ns = t_ns - start_t_ns;

    vio_t_ns.emplace_back(t_ns);
    vio_t_w_i.emplace_back(data->T_w_i.translation());

    Sophus::SE3d T_w_i = data->T_w_i;
    Eigen::Vector3d vel_w_i = data->vel_w_i;
    Eigen::Vector3d bg = data->bias_gyro;
    Eigen::Vector3d ba = data->bias_accel;

    vector<float> vals;
    vals.push_back(since_start_ns * 1e-9);
    for (int i = 0; i < 3; i++) vals.push_back(vel_w_i[i]);
    for (int i = 0; i < 3; i++) vals.push_back(T_w_i.translation()[i]);
    for (int i = 0; i < 3; i++) vals.push_back(bg[i]);
    for (int i = 0; i < 3; i++) vals.push_back(ba[i]);

    vio_data_log.Log(vals);
  }

  pangolin::Plotter *plotter;
  pangolin::DataLog imu_data_log{};
  Calibration<double> calib;
  VioConfig config;
  void start_ui(const Sophus::SE3d &T_w_i_init, const Calibration<double> &cal, const VioConfig &conf) {
    ui_runner_thread = thread(&slam_tracker_ui::ui_runner, this, T_w_i_init, cal, conf);
  }

  pangolin::Var<bool> follow{"ui.follow", true, false, true};
  pangolin::Var<bool> show_est_pos{"ui.show_est_pos", true, false, true};
  pangolin::Var<bool> show_est_vel{"ui.show_est_vel", false, false, true};
  pangolin::Var<bool> show_est_bg{"ui.show_est_bg", false, false, true};
  pangolin::Var<bool> show_est_ba{"ui.show_est_ba", false, false, true};

  void ui_runner(const Sophus::SE3d &T_w_i_init, const Calibration<double> &cal, const VioConfig &conf) {
    constexpr int UI_WIDTH = 200;

    calib = cal;
    config = conf;
    string window_name = "Basalt 6DoF Tracker for Monado";
    pangolin::CreateWindowAndBind(window_name, 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    pangolin::View &img_view_display = pangolin::CreateDisplay()
                                           .SetBounds(0.4, 1.0, pangolin::Attach::Pix(UI_WIDTH), 0.4)
                                           .SetLayout(pangolin::LayoutEqual);

    pangolin::View &plot_display = pangolin::CreateDisplay().SetBounds(0.0, 0.4, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    plotter = new pangolin::Plotter(&imu_data_log, 0.0, 100, -3.0, 3.0, 0.01, 0.01);
    plot_display.AddDisplay(*plotter);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < calib.intrinsics.size()) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);
      iv->UseNN() = true;  // Disable antialiasing (toggle it back with the N key)

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function = [idx, this](auto &v) { this->draw_image_overlay(v, idx); };
    }

    Eigen::Vector3d cam_p(0.5, -2, -2);
    cam_p = T_w_i_init.so3() * calib.T_i_c[0].so3() * cam_p;
    cam_p[2] = 1;

    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(cam_p[0], cam_p[1], cam_p[2], 0, 0, 0, pangolin::AxisZ));

    pangolin::View &display3D = pangolin::CreateDisplay()
                                    .SetAspect(-640 / 480.0)
                                    .SetBounds(0.4, 1.0, 0.4, 1.0)
                                    .SetHandler(new pangolin::Handler3D(camera));

    while (running && !pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (follow) {
        // TODO: There is a small race condition here over
        // curr_vis_data that is also present in the original basalt examples
        if (curr_vis_data) {
          auto T_w_i = curr_vis_data->states.back();
          T_w_i.so3() = Sophus::SO3d();

          camera.Follow(T_w_i.matrix());
        }
      }

      display3D.Activate(camera);
      glClearColor(1.0, 1.0, 1.0, 1.0);

      draw_scene();

      img_view_display.Activate();

      if (fixed_depth.GuiChanged() && opt_flow_depth_queue != nullptr) {
        opt_flow_depth_queue->push(fixed_depth);
      }
      depth_guess = opt_flow->depth_guess;

      {
        pangolin::GlPixFormat fmt;
        fmt.glformat = GL_LUMINANCE;
        fmt.gltype = GL_UNSIGNED_SHORT;
        fmt.scalable_internal_format = GL_LUMINANCE16;

        if (curr_vis_data && curr_vis_data->opt_flow_res && curr_vis_data->opt_flow_res->input_images) {
          auto &img_data = curr_vis_data->opt_flow_res->input_images->img_data;

          for (size_t cam_id = 0; cam_id < num_cams; cam_id++) {
            if (img_data[cam_id].img) {
              img_view[cam_id]->SetImage(img_data[cam_id].img->ptr, img_data[cam_id].img->w, img_data[cam_id].img->h,
                                         img_data[cam_id].img->pitch, fmt);
            }
          }
        }

        draw_plots();
      }

      if (show_est_vel.GuiChanged() || show_est_pos.GuiChanged() || show_est_ba.GuiChanged() ||
          show_est_bg.GuiChanged()) {
        draw_plots();
      }

      pangolin::FinishFrame();
    }

    pangolin::QuitAll();
    cout << "Finished ui_runner\n";
  }

  pangolin::Var<int> show_frame{"ui.show_frame", 0, pangolin::META_FLAG_READONLY};
  pangolin::Var<bool> show_flow{"ui.show_flow", false, false, true};
  pangolin::Var<bool> show_tracking_guess{"ui.show_tracking_guess", false, false, true};
  pangolin::Var<bool> show_matching_guess{"ui.show_matching_guess", false, false, true};
  pangolin::Var<bool> show_obs{"ui.show_obs", true, false, true};
  pangolin::Var<bool> show_ids{"ui.show_ids", false, false, true};
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

  void draw_image_overlay(pangolin::View &v, size_t cam_id) {
    UNUSED(v);

    if (!curr_vis_data ||                                               //
        !curr_vis_data->opt_flow_res ||                                 //
        !curr_vis_data->opt_flow_res->input_images ||                   //
        curr_vis_data->opt_flow_res->input_images->img_data.empty() ||  //
        !curr_vis_data->opt_flow_res->input_images->img_data.at(0).img) {
      return;
    }
    const auto &frames = curr_vis_data->opt_flow_res->input_images->img_data;
    const auto &cam0_img = frames.at(0).img;

    size_t NUM_CAMS = curr_vis_data->projections->size();

    if (show_obs) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 0.0);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      if (curr_vis_data && cam_id < NUM_CAMS) {
        const auto &points = curr_vis_data->projections->at(cam_id);

        if (!points.empty()) {
          double min_id = points[0][2];
          double max_id = points[0][2];

          for (const auto &points2 : *curr_vis_data->projections) {
            for (const auto &p : points2) {
              min_id = std::min(min_id, p[2]);
              max_id = std::max(max_id, p[2]);
            }
          }

          for (const auto &c : points) {
            double u = c[0];
            double v = c[1];
            double depth = c[2] != 0.0 ? 1.0 / c[2] : std::numeric_limits<double>::infinity();
            int id = c[3];

            double width = cam0_img->w;
            double unit_radius = width / 96;
            double radius = unit_radius / depth;

            double min_depth = 1.0 / 3;  // 1/3 comes from how valid_kp is computed in sqrt_keypoint_vio.cpp
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
          for (const auto &cam_projs : *curr_vis_data->projections) {
            for (const Vector4d &v : cam_projs) avg_invdepth += v.z();
            num_features += cam_projs.size();
          }
          bool valid = avg_invdepth > 0 && num_features > 0;
          float default_depth = config.optical_flow_matching_default_depth;
          double avg_depth = valid ? num_features / avg_invdepth : default_depth;

          for (const Vector4d &kp1 : keypoints1) {
            double u1 = kp1.x();
            double v1 = kp1.y();
            // double invdist1 = kp1.z();
            double id1 = kp1.w();

            double u0 = 0;
            double v0 = 0;
            bool found = false;
            for (const Vector4d &kp0 : keypoints0) {
              double id0 = kp0.w();
              if (id1 != id0) continue;
              u0 = kp0.x();
              v0 = kp0.y();
              found = true;
              break;
            }

            if (found) {
              if (show_same_pixel_guess) {
                glColor3f(0, 1, 1);  // Cyan
                pangolin::glDrawLine(u1, v1, u0, v0);
              }

              if (show_reproj_fix_depth_guess) {
                glColor3f(1, 1, 0);  // Yellow
                auto off = calib.viewOffset({u0, v0}, fixed_depth, 0, cam_id);
                pangolin::glDrawLine(u1, v1, u0 - off.x(), v0 - off.y());
              }

              if (show_reproj_avg_depth_guess) {
                glColor3f(1, 0, 1);  // Magenta
                auto off = calib.viewOffset({u0, v0}, avg_depth, 0, cam_id);
                pangolin::glDrawLine(u1, v1, u0 - off.x(), v0 - off.y());
              }

              if (show_active_guess) {
                glColor3f(1, 0, 0);  // Red
                Vector2d off{0, 0};
                if (config.optical_flow_matching_guess_type != MatchingGuessType::SAME_PIXEL) {
                  off = calib.viewOffset({u0, v0}, opt_flow->depth_guess, 0, cam_id);
                }
                pangolin::glDrawLine(u1, v1, u0 - off.x(), v0 - off.y());
              }
            }
          }
        }

        glColor3f(0.0, 1.0, 0.0);
        pangolin::GlFont::I().Text("Tracked %d points", points.size()).Draw(5, 20);
      }
    }

    if (show_flow) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 0.0);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      auto &keypoints = curr_vis_data->opt_flow_res->keypoints;

      if (keypoints.size() > 0) {
        const Eigen::aligned_map<basalt::KeypointId, Keypoint> &kp_map = keypoints[cam_id];

        for (const auto &kv : kp_map) {
          Eigen::MatrixXf transformed_patch = kv.second.pose.linear() * opt_flow->patch_coord;
          transformed_patch.colwise() += kv.second.pose.translation();

          for (int i = 0; i < transformed_patch.cols(); i++) {
            const Eigen::Vector2f c = transformed_patch.col(i);
            pangolin::glDrawCirclePerimeter(c[0], c[1], 0.5f);
          }

          const Eigen::Vector2f c = kv.second.pose.translation();

          if (show_ids) pangolin::GlFont::I().Text("%d", kv.first).Draw(5 + c[0], 5 + c[1]);
        }

        pangolin::GlFont::I().Text("Detected %d keypoints", kp_map.size()).Draw(5, 40);
      }
    }

    if (show_tracking_guess) {
      size_t frame_id = show_frame;
      if (frame_id < 1) goto out_show_tracking_guess;

      auto new_kpts = curr_vis_data->opt_flow_res->keypoints[cam_id];
      auto prev_kpts = prev_vis_data->opt_flow_res->keypoints[cam_id];
      auto guess_obs = curr_vis_data->opt_flow_res->tracking_guesses[cam_id];

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
      for (auto &[kpid, kpt] : new_kpts) {
        if (prev_kpts.count(kpid) == 0) continue;
        if (guess_obs.count(kpid) == 0) continue;

        auto n = kpt.pose.translation();
        auto p = prev_kpts.at(kpid).pose.translation();
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
      for (auto &[kpid, kpt] : new_kpts) {
        if (cam0_kpts.count(kpid) == 0) continue;
        if (guess_obs.count(kpid) == 0) continue;

        auto n = kpt.pose.translation();
        auto c = cam0_kpts.at(kpid).pose.translation();
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
      for (const Rect &m : curr_vis_data->opt_flow_res->input_images->masks[cam_id].masks) {
        pangolin::glDrawRect(m.x, m.y, m.x + m.w, m.y + m.h);
      }
    }

    int C = config.optical_flow_detection_grid_size;

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
      auto drawPoint = [this, &points, w, h](float u, float v, int j, bool draw_c0_uv) {
        Vector2d ci_uv{u, v};
        Vector2d c0_uv;
        double _;
        bool projected = calib.projectBetweenCams(ci_uv, opt_flow->depth_guess, c0_uv, _, j, 0);
        bool in_bounds = c0_uv.x() >= 0 && c0_uv.x() < w && c0_uv.y() >= 0 && c0_uv.y() < h;
        bool valid = projected && in_bounds;

        // Define color
        GLfloat invalid_color[4] = {1, 0, 0, 0.5};      // red
        GLfloat in_bounds_color[4] = {1, 0.5, 0, 0.5};  // orange
        GLfloat projected_color[4] = {1, 0.9, 0, 0.5};  // yellow
        GLfloat valid_color[4] = {0, 1, 0, 0.5};        // green
        GLfloat *color = invalid_color;
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

  void draw_scene() {
    glPointSize(3);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glColor3ubv(cam_color);
    Eigen::aligned_vector<Eigen::Vector3d> sub_gt(vio_t_w_i.begin(), vio_t_w_i.end());
    pangolin::glDrawLineStrip(sub_gt);

    if (curr_vis_data) {
      for (const auto &p : curr_vis_data->states)
        for (const auto &t_i_c : calib.T_i_c) render_camera((p * t_i_c).matrix(), 2.0, state_color, 0.1);

      for (const auto &p : curr_vis_data->frames)
        for (const auto &t_i_c : calib.T_i_c) render_camera((p * t_i_c).matrix(), 2.0, pose_color, 0.1);

      for (const auto &t_i_c : calib.T_i_c)
        render_camera((curr_vis_data->states.back() * t_i_c).matrix(), 2.0, cam_color, 0.1);

      glColor3ubv(pose_color);
      pangolin::glDrawPoints(curr_vis_data->points);
    }

    pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);
  }

  OpticalFlowInput::Ptr last_img_data{};
  void update_last_image(OpticalFlowInput::Ptr &data) { last_img_data = data; }
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
      plotter->AddSeries("$0", "$8", pangolin::DrawingModeLine, pangolin::Colour::Green(), "gyro bias y",
                         &vio_data_log);
      plotter->AddSeries("$0", "$9", pangolin::DrawingModeLine, pangolin::Colour::Blue(), "gyro bias z", &vio_data_log);
    }

    if (show_est_ba) {
      plotter->AddSeries("$0", "$10", pangolin::DrawingModeLine, pangolin::Colour::Red(), "accel bias x",
                         &vio_data_log);
      plotter->AddSeries("$0", "$11", pangolin::DrawingModeLine, pangolin::Colour::Green(), "accel bias y",
                         &vio_data_log);
      plotter->AddSeries("$0", "$12", pangolin::DrawingModeLine, pangolin::Colour::Blue(), "accel bias z",
                         &vio_data_log);
    }

    if (last_img_data) {
      double t = last_img_data->t_ns * 1e-9;
      plotter->AddMarker(pangolin::Marker::Vertical, t, pangolin::Marker::Equal, pangolin::Colour::White());
    }
  }
};
}  // namespace xrt::auxiliary::tracking::slam
