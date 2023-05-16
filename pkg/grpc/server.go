package grpc

import (
	"context"
	"log"
	"net"
	"time"

	"github.com/extrame/llama.go/pkg/llama"
	"github.com/extrame/llama.go/pkg/ml"
	"github.com/pkg/errors"
	"google.golang.org/grpc"
)

type Server struct {
	LlamaGoServiceServer
	baseCtx     context.Context
	RunningPods int64
	MaxPods     int64
	Vocab       *ml.Vocab
	Model       *llama.Model
	Params      *llama.ModelParams
}

//下行通道
func (s *Server) Do(job *Job, server LlamaGoService_DoServer) error {
	return s.do(job, server)
}

type JobStub struct {
	client      LlamaGoService_DoServer
	JobCancelFn context.CancelFunc
}

func (c *JobStub) SendResponse(cmd *Output) error {
	var errC = make(chan error)
	go func() {
		errC <- c.client.Send(cmd)
	}()
	select {
	case err := <-errC:
		if err != nil {
			c.JobCancelFn()
		}
		return err
	case <-time.After(5 * time.Second):
		c.JobCancelFn()
		return errors.New("timeout")
	}
}

//main 从sub获取数据
func NewServer(addr string, pods int64, vocab *ml.Vocab, model *llama.Model, params *llama.ModelParams, ctx context.Context) (server *Server, err error) {
	//创建grpc服务器
	var lis net.Listener
	lis, err = net.Listen("tcp", addr)
	if err == nil {
		s := grpc.NewServer()
		server = &Server{
			baseCtx: ctx,
			MaxPods: pods,
			Vocab:   vocab,
			Model:   model,
		}

		RegisterLlamaGoServiceServer(s, server)
		go func() {
			err := s.Serve(lis)
			if err != nil {
				log.Fatal(err)
			}
		}()
		go func() {
			<-ctx.Done()
			s.GracefulStop()
		}()
	}
	return
}
