package grpc

import (
	"container/ring"
	"runtime"
	"sync/atomic"
	"time"

	"github.com/extrame/llama.go/pkg/llama"
	"github.com/extrame/llama.go/pkg/ml"
	"github.com/extrame/llama.go/pkg/utils"
)

func (s *Server) do(j *Job, server LlamaGoService_DoServer) error {

	defer atomic.AddInt64(&s.RunningPods, -1)
	defer runtime.GC()

	// TODO: Proper logging
	// fmt.Printf("\n[ PROCESSING ] Starting job # %s", jobID)

	prompt := " " + j.Prompt // add a space to match LLaMA tokenizer behavior

	// tokenize the prompt
	embdPrompt := ml.Tokenize(s.Vocab, prompt, true)

	// ring buffer for last N tokens
	lastNTokens := ring.New(int(s.Params.CtxSize))

	// method to append a token to the ring buffer
	appendToken := func(token uint32) {
		lastNTokens.Value = token
		lastNTokens = lastNTokens.Next()
	}

	// zeroing the ring buffer
	for i := 0; i < int(s.Params.CtxSize); i++ {
		appendToken(0)
	}

	evalCounter := 0
	tokenCounter := 0
	pastCount := uint32(0)
	consumedCount := uint32(0)             // number of tokens, already processed from the user prompt
	remainedCount := s.Params.PredictCount // how many tokens we still need to generate to achieve predictCount
	embd := make([]uint32, 0, s.Params.BatchSize)
	evalPerformance := make([]int64, 0, s.Params.PredictCount)
	samplePerformance := make([]int64, 0, s.Params.PredictCount)
	fullPerformance := make([]int64, 0, s.Params.PredictCount)

	// new context opens sync channel and starts workers for tensor compute
	ctx := llama.NewContext(s.Model, s.Params)

	for remainedCount > 0 {

		// TODO: Store total time of evaluation and average per token + token count
		start := time.Now().UnixNano()

		if len(embd) > 0 {

			// infinite text generation via context swapping
			// if we run out of context:
			// - take the n_keep first tokens from the original prompt (via n_past)
			// - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch

			if pastCount+uint32(len(embd)) > s.Params.CtxSize {
				leftCount := pastCount - s.Params.KeepCount
				pastCount = s.Params.KeepCount

				// insert n_left/2 tokens at the start of embd from last_n_tokens
				// embd = append(lastNTokens[:leftCount/2], embd...)
				embd = append(llama.ExtractTokens(lastNTokens.Move(-int(leftCount/2)), int(leftCount/2)), embd...)
			}

			evalStart := time.Now().UnixNano()
			if err := llama.Eval(ctx, s.Vocab, s.Model, embd, pastCount, s.Params); err != nil {
				// TODO: Finish job properly with [failed] status
				server.Send(&Output{
					Status: Status_FAILED,
					Output: err.Error(),
				})
			}
			evalPerformance = append(evalPerformance, time.Now().UnixNano()-evalStart)
			evalCounter++
		}

		pastCount += uint32(len(embd))
		embd = embd[:0]

		if int(consumedCount) < len(embdPrompt) {

			for len(embdPrompt) > int(consumedCount) && len(embd) < int(s.Params.BatchSize) {

				embd = append(embd, embdPrompt[consumedCount])
				appendToken(embdPrompt[consumedCount])
				consumedCount++
			}

		} else {

			//if s.Params.IgnoreEOS {
			//	Ctx.Logits[ml.TOKEN_EOS] = 0
			//}

			sampleStart := time.Now().UnixNano()
			id := llama.SampleTopPTopK( /*ctx,*/ ctx.Logits,
				lastNTokens, s.Params.RepeatLastN,
				s.Params.TopK, s.Params.TopP,
				s.Params.Temp, s.Params.RepeatPenalty)
			samplePerformance = append(samplePerformance, time.Now().UnixNano()-sampleStart)

			appendToken(id)

			// replace end of text token with newline token when in interactive mode
			//if id == ml.TOKEN_EOS && Params.Interactive && !Params.Instruct {
			//	id = ml.NewLineToken
			//}

			embd = append(embd, id) // add to the context

			remainedCount-- // decrement remaining sampling budget
		}

		fullPerformance = append(fullPerformance, time.Now().UnixNano()-start)

		// skip adding the whole prompt to the output if processed at once
		if evalCounter == 0 && int(consumedCount) == len(embdPrompt) {
			continue
		}

		// --- assemble the final ouptut, EXCLUDING the prompt

		for _, id := range embd {

			tokenCounter++
			token := ml.Token2Str(s.Vocab, id) // TODO: Simplify

			server.Send(&Output{
				Status: Status_RUNNING,
				Output: token,
			})
		}
	}

	// close sync channel and stop compute workers
	ctx.ReleaseContext()

	//if ml.DEBUG {
	utils.Colorize("\n\n=== EVAL TIME | ms ===\n\n")
	for _, time := range evalPerformance {
		utils.Colorize("%d | ", time/1_000_000)
	}

	utils.Colorize("\n\n=== SAMPLING TIME | ms ===\n\n")
	for _, time := range samplePerformance {
		utils.Colorize("%d | ", time/1_000_000)
	}

	utils.Colorize("\n\n=== FULL TIME | ms ===\n\n")
	for _, time := range fullPerformance {
		utils.Colorize("%d | ", time/1_000_000)
	}

	avgEval := int64(0)
	for _, time := range fullPerformance {
		avgEval += time / 1_000_000
	}
	avgEval /= int64(len(fullPerformance))

	utils.Colorize(
		"\n\n[light_magenta][ HALT ][white] Time per token: [light_cyan]%d[white] ms | Tokens per second: [light_cyan]%.2f\n\n",
		avgEval,
		float64(1000)/float64(avgEval))
	//}

	// TODO: Proper logging
	// fmt.Printf("\n[ PROCESSING ] Finishing job # %s", jobID)
	return nil
}
