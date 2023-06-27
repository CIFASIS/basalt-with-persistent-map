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

    for (const auto& [kpt_id, kpt] : lmdb_.getLandmarks()) {
      kp2.push_back(kpt_id);
      descr2.push_back(kpt.descriptor);
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
        curr_frame->keypoints.at(i).erase(kp_id);
        num_matches_++;
      }
    }
  }
}
}  // namespace basalt
