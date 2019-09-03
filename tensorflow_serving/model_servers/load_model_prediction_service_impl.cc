/* Copyright 2018 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <string>

#include "tensorflow_serving/model_servers/load_model_prediction_service_impl.h"

#include "grpc/grpc.h"
#include "tensorflow_serving/model_servers/grpc_status_util.h"
#include "tensorflow_serving/servables/tensorflow/get_model_metadata_impl.h"

namespace tensorflow {
namespace serving {

namespace {

int DeadlineToTimeoutMillis(const gpr_timespec deadline) {
  return gpr_time_to_millis(
      gpr_time_sub(gpr_convert_clock_type(deadline, GPR_CLOCK_MONOTONIC),
                   gpr_now(GPR_CLOCK_MONOTONIC)));
}

}  // namespace

::grpc::Status LoadModelPredictionServiceImpl::LoadPredict(::grpc::ServerContext *context,
                                              const PredictRequest *request,
                                              PredictResponse *response) {
  tensorflow::RunOptions run_options = tensorflow::RunOptions();
  if (enforce_session_run_timeout_) {
    run_options.set_timeout_in_ms(
        DeadlineToTimeoutMillis(context->raw_deadline()));
  }
  ModelSpec m_spec = request->model_spec();
  string m_name = m_spec.name();
	LOG(INFO) << "Load Model " + m_name;
  /*
  ModelServerConfig BuildSingleModelConfig(const string& model_name,
                                           const string& model_base_path) {
    ModelServerConfig config;
    LOG(INFO) << "Building single TensorFlow model file config: "
              << " model_name: " << model_name
              << " model_base_path: " << model_base_path;
    tensorflow::serving::ModelConfig* single_model =
        config.mutable_model_config_list()->add_config();
    single_model->set_name(model_name);
    single_model->set_base_path(model_base_path);
    single_model->set_model_platform(
        tensorflow::serving::kTensorFlowModelPlatform);
    return config;
  } */

  //string m_path = "/data/models/";
  //string model_platform = "tensorflow";
  //ModelConfigList mconfig = [
  //  m_name,
  //  m_path,
  //  model_platform
  //];

  const ::grpc::Status status =
      ToGRPCStatus(predictor_->Predict(run_options, core_, *request, response));

  if (!status.ok()) {
    VLOG(1) << "Predict failed: " << status.error_message();
  }
  return status;
}

::grpc::Status LoadModelPredictionServiceImpl::GetModelMetadata(
    ::grpc::ServerContext *context, const GetModelMetadataRequest *request,
    GetModelMetadataResponse *response) {
  if (!use_saved_model_) {
    return ToGRPCStatus(
        errors::InvalidArgument("GetModelMetadata API is only available when "
                                "use_saved_model is set to true"));
  }
  const ::grpc::Status status = ToGRPCStatus(
      GetModelMetadataImpl::GetModelMetadata(core_, *request, response));
  if (!status.ok()) {
    VLOG(1) << "GetModelMetadata failed: " << status.error_message();
  }
  return status;
}

}  // namespace serving
}  // namespace tensorflow
