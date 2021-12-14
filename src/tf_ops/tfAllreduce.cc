#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "allreduce.hxx"

using namespace tensorflow;

#define EIGEN_USE_THREADS

#define ALLREDUCE_APPLY_OPBUILDER(NAME) \
REGISTER_OP(NAME) \
    .Attr("T: {int32, int64, float32, float64}") \
    .Attr("gaspi_allreduce : int") \
    .Input("input_tensor: T") \
    .Output("output_tensor: T") \
    .SetShapeFn([](shape_inference::InferenceContext* c) { \
    c->set_output(0, c->input(0)); \
    return Status::OK(); \
    }) \
    .Doc(R"doc( Gaspi allreduce )doc");
template<template<class> class AllreduceTYPE, typename T>
class AllreduceApply : public OpKernel {
public:
    explicit AllreduceApply(OpKernelConstruction* context)
            : OpKernel(context) {
        tensorflow::int64 ptr;

        OP_REQUIRES_OK(context, context->GetAttr("gaspi_allreduce", &ptr));
        gaspi_allreduce = reinterpret_cast<AllreduceTYPE<T>* >(ptr);
    }

    void Compute(OpKernelContext* context) override {
        auto input_tensor = context->input(0);
        gaspi_allreduce->apply(input_tensor.flat<T>().data(), input_tensor.NumElements());
        context->set_output(0, input_tensor);
    }
private:
    AllreduceTYPE<T>* gaspi_allreduce;
};

#define ALLREDUCE_FLUSH_OPBUILDER(NAME) \
REGISTER_OP(NAME) \
    .Attr("T: {int32, int64, float32, float64}") \
    .Attr("gaspi_allreduce : int") \
    .Input("input_tensor: T") \
    .Output("output_tensor: T") \
    .SetShapeFn([](shape_inference::InferenceContext* c) { \
      c->set_output(0, c->input(0)); \
      return Status::OK();\
    }) \
    .Doc(R"doc( Gaspi allreduce )doc");
template<template<class> class AllreduceTYPE, typename T>
class AllreduceFlush : public OpKernel {
public:
    explicit AllreduceFlush(OpKernelConstruction* context)
            : OpKernel(context) {
        tensorflow::int64 ptr;

        OP_REQUIRES_OK(context, context->GetAttr("gaspi_allreduce", &ptr));
        gaspi_allreduce = reinterpret_cast<AllreduceTYPE<T>* >(ptr);
    }

    void Compute(OpKernelContext* context) override {
        auto input_tensor = context->input(0); 
        gaspi_allreduce->flush(input_tensor.flat<T>().data(), input_tensor.NumElements());
        context->set_output(0, input_tensor);
    }
private:
    AllreduceTYPE<T>* gaspi_allreduce;
};

// Register OPS
////////////////////////////////////

#define ALLREDUCE_APPLY_CLASS(NAME,ALLREDUCE) \
template<typename T> \
class NAME : public AllreduceApply<ALLREDUCE, T> { \
public: \
    NAME(OpKernelConstruction* context) : AllreduceApply<ALLREDUCE, T>(context) {}; \
};

#define ALLREDUCE_FLUSH_CLASS(NAME,ALLREDUCE) \
template<typename T> \
class NAME : public AllreduceFlush<ALLREDUCE, T> { \
public: \
    NAME(OpKernelConstruction* context) : AllreduceFlush<ALLREDUCE, T>(context) {}; \
};

// AllToOneAllreduce
ALLREDUCE_APPLY_OPBUILDER("AlltooneAllreduceApply");
ALLREDUCE_APPLY_CLASS(AlltooneAllreduceApply, AllToOneAllreduce);

// Comprex_AllToOneAllreduce
ALLREDUCE_APPLY_OPBUILDER("ComprexAlltooneAllreduceApply");
ALLREDUCE_APPLY_CLASS(ComprexAlltooneAllreduceApply, Comprex_AllToOneAllreduce);
ALLREDUCE_FLUSH_OPBUILDER("ComprexAlltooneAllreduceFlush");
ALLREDUCE_FLUSH_CLASS(ComprexAlltooneAllreduceFlush, Comprex_AllToOneAllreduce);

// RingAllreduce
ALLREDUCE_APPLY_OPBUILDER("RingAllreduceApply");
ALLREDUCE_APPLY_CLASS(RingAllreduceApply, RingAllreduce);

// Comprex_RingAllreduce
ALLREDUCE_APPLY_OPBUILDER("ComprexRingAllreduceApply");
ALLREDUCE_APPLY_CLASS(ComprexRingAllreduceApply, Comprex_RingAllreduce);
ALLREDUCE_FLUSH_OPBUILDER("ComprexRingAllreduceFlush");
ALLREDUCE_FLUSH_CLASS(ComprexRingAllreduceFlush, Comprex_RingAllreduce);

// BigRingAllreduce
ALLREDUCE_APPLY_OPBUILDER("BigringAllreduceApply");
ALLREDUCE_APPLY_CLASS(BigringAllreduceApply, BigRingAllreduce);

// Comprex_BigRingAllreduce
ALLREDUCE_APPLY_OPBUILDER("ComprexBigringAllreduceApply");
ALLREDUCE_APPLY_CLASS(ComprexBigringAllreduceApply, Comprex_BigRingAllreduce)
ALLREDUCE_FLUSH_OPBUILDER("ComprexBigringAllreduceFlush");
ALLREDUCE_FLUSH_CLASS(ComprexBigringAllreduceFlush, Comprex_BigRingAllreduce)


// Create Kernel
/////////////////////////////////////
#define ALLREDUCE_KERNELBUILDER(NAME,ALLREDUCE,TYPE) \
    REGISTER_KERNEL_BUILDER( \
        Name(NAME) \
        .Device(DEVICE_CPU) \
        .TypeConstraint<TYPE>("T"), \
        ALLREDUCE<TYPE>);

//AllToOneAllreduce
ALLREDUCE_KERNELBUILDER("AlltooneAllreduceApply",AlltooneAllreduceApply,float);

// Comprex_AllToOneAllreduce
ALLREDUCE_KERNELBUILDER("ComprexAlltooneAllreduceApply",ComprexAlltooneAllreduceApply,float);
ALLREDUCE_KERNELBUILDER("ComprexAlltooneAllreduceFlush",ComprexAlltooneAllreduceFlush,float);

//RingAllreduce
ALLREDUCE_KERNELBUILDER("RingAllreduceApply",RingAllreduceApply,float);

// Comprex_RingAllreduce
ALLREDUCE_KERNELBUILDER("ComprexRingAllreduceApply",ComprexRingAllreduceApply,float);
ALLREDUCE_KERNELBUILDER("ComprexRingAllreduceFlush",ComprexRingAllreduceFlush,float);

//BigRingAllreduce
ALLREDUCE_KERNELBUILDER("BigringAllreduceApply",BigringAllreduceApply,float);

// Comprex_BigRingAllreduce
ALLREDUCE_KERNELBUILDER("ComprexBigringAllreduceApply",ComprexBigringAllreduceApply,float);
ALLREDUCE_KERNELBUILDER("ComprexBigringAllreduceFlush",ComprexBigringAllreduceFlush,float);
