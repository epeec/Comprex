#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

#define EIGEN_USE_THREADS

class GaspiAllreduceOp : public AsyncOpKernel {
public:
  explicit GaspiAllreduceOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, Status::OK(), done); // TODO: check if GPI is initiallized here!

    auto node_name = name();
    //auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, tensor.shape(), &output), done);
    // ReadyEvent makes sure input tensor is ready, and output is allocated.
    // auto ready_event = std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
    // auto hvd_context = std::make_shared<TFOpContext>(context);
    // auto hvd_tensor = std::make_shared<TFTensor>(tensor);
    // auto hvd_output = std::make_shared<TFTensor>(*output);
    /* auto enqueue_result = EnqueueTensorAllreduce(
        hvd_context, hvd_tensor, hvd_output, ready_event, node_name, device,
        [context, done](const common::Status& status) {
          context->SetStatus(ConvertStatus(status));
          done();
        });*/
    OP_REQUIRES_OK_ASYNC(context, Status::OK(), done); // TODO: check here if GPI returned successfully
    done();
  }
};

REGISTER_KERNEL_BUILDER(Name("GaspiAllreduce").Device(DEVICE_CPU),
                        GaspiAllreduceOp);

REGISTER_OP("GaspiAllreduce")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc( Gaspi allreduce )doc");