// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.2.0
// - protoc             v3.20.1--rc1
// source: pkg/grpc/message.proto

package grpc

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.32.0 or later.
const _ = grpc.SupportPackageIsVersion7

// LlamaGoServiceClient is the client API for LlamaGoService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type LlamaGoServiceClient interface {
	Do(ctx context.Context, in *Job, opts ...grpc.CallOption) (LlamaGoService_DoClient, error)
}

type llamaGoServiceClient struct {
	cc grpc.ClientConnInterface
}

func NewLlamaGoServiceClient(cc grpc.ClientConnInterface) LlamaGoServiceClient {
	return &llamaGoServiceClient{cc}
}

func (c *llamaGoServiceClient) Do(ctx context.Context, in *Job, opts ...grpc.CallOption) (LlamaGoService_DoClient, error) {
	stream, err := c.cc.NewStream(ctx, &LlamaGoService_ServiceDesc.Streams[0], "/module.LlamaGoService/Do", opts...)
	if err != nil {
		return nil, err
	}
	x := &llamaGoServiceDoClient{stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

type LlamaGoService_DoClient interface {
	Recv() (*Output, error)
	grpc.ClientStream
}

type llamaGoServiceDoClient struct {
	grpc.ClientStream
}

func (x *llamaGoServiceDoClient) Recv() (*Output, error) {
	m := new(Output)
	if err := x.ClientStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

// LlamaGoServiceServer is the server API for LlamaGoService service.
// All implementations must embed UnimplementedLlamaGoServiceServer
// for forward compatibility
type LlamaGoServiceServer interface {
	Do(*Job, LlamaGoService_DoServer) error
	mustEmbedUnimplementedLlamaGoServiceServer()
}

// UnimplementedLlamaGoServiceServer must be embedded to have forward compatible implementations.
type UnimplementedLlamaGoServiceServer struct {
}

func (UnimplementedLlamaGoServiceServer) Do(*Job, LlamaGoService_DoServer) error {
	return status.Errorf(codes.Unimplemented, "method Do not implemented")
}
func (UnimplementedLlamaGoServiceServer) mustEmbedUnimplementedLlamaGoServiceServer() {}

// UnsafeLlamaGoServiceServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to LlamaGoServiceServer will
// result in compilation errors.
type UnsafeLlamaGoServiceServer interface {
	mustEmbedUnimplementedLlamaGoServiceServer()
}

func RegisterLlamaGoServiceServer(s grpc.ServiceRegistrar, srv LlamaGoServiceServer) {
	s.RegisterService(&LlamaGoService_ServiceDesc, srv)
}

func _LlamaGoService_Do_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(Job)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(LlamaGoServiceServer).Do(m, &llamaGoServiceDoServer{stream})
}

type LlamaGoService_DoServer interface {
	Send(*Output) error
	grpc.ServerStream
}

type llamaGoServiceDoServer struct {
	grpc.ServerStream
}

func (x *llamaGoServiceDoServer) Send(m *Output) error {
	return x.ServerStream.SendMsg(m)
}

// LlamaGoService_ServiceDesc is the grpc.ServiceDesc for LlamaGoService service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var LlamaGoService_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "module.LlamaGoService",
	HandlerType: (*LlamaGoServiceServer)(nil),
	Methods:     []grpc.MethodDesc{},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "Do",
			Handler:       _LlamaGoService_Do_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "pkg/grpc/message.proto",
}
