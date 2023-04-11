package main

import (
	"fmt"
	"os"
	"runtime"
	"strings"

	// "github.com/mattn/go-colorable"
	flags "github.com/jessevdk/go-flags"
	colorable "github.com/mattn/go-colorable"
	"github.com/mitchellh/colorstring"

	"github.com/gotzmann/llama.go/llama"
	"github.com/gotzmann/llama.go/ml"
)

type ModelParams struct {
	seed         int    // -1 // RNG seed
	threadsCount int    // min(4, std::thread::hardware_concurrency())
	predictCount uint32 // 128 // new tokens to predict
	repeatLastN  uint32 // 64 // last n tokens to penalize
	partsCount   int    // -1 // amount of model parts (-1 = determine from model dimensions)
	ctxSize      uint32 // 512 // context size
	batchSize    uint32 // 8 // batch size for prompt processing
	keepCount    uint32

	// --- sampling parameters

	topK          uint32  // 40
	topP          float32 // 0.95
	temp          float32 // 0.80
	repeatPenalty float32 // 1.10

	model       string // model path
	prompt      string // ""
	inputPrefix string // "" // string to prefix user inputs with

	antiprompt []string // string upon seeing which more user input is prompted

	memoryFP16   bool // true // use f16 instead of f32 for memory kv
	randomPrompt bool // false // do not randomize prompt if none provided
	useColor     bool // false // use color to distinguish generations and inputs
	interactive  bool // false // interactive mode

	embedding        bool // false // get only sentence embedding
	interactiveStart bool // false // wait for user input immediately

	instruct   bool // false // instruction mode (used for Alpaca models)
	ignoreEOS  bool // false // do not stop generating after eos
	perplexity bool // false // compute perplexity over the prompt
	use_mlock  bool // false // use mlock to keep model in memory
	memTest    bool // false // compute maximum memory usage

	verbosePrompt bool
}

func main() {

	//defer profile.Start(profile.ProfilePath(".")).Stop()

	// --- Parse command line args and set default parameters

	var opts struct {
		Prompt  string  `long:"prompt" description:"Text prompt from user to feed the model input"`
		Model   string  `long:"model" description:"Path and file name of converted .bin LLaMA model"`
		Threads int     `long:"threads" description:"Adjust to the number of CPU cores you want to use [ all cores by default ]"`
		Predict uint32  `long:"predict" description:"Number of tokens to predict [ 128 by default ]"`
		Context uint32  `long:"context" description:"Context size in tokens [ 512 by default ]"`
		Temp    float32 `long:"temp" description:"Model temperature hyper parameter [ 0.8 by default ]"`
		Silent  bool    `long:"silent" description:"Hide welcome logo and other output [ show by default ]"`
		Chat    bool    `long:"chat" description:"Chat with user in interactive mode instead of compute over static prompt"`
	}

	flags.Parse(&opts)

	prompt := opts.Prompt

	// TODO Optimize default settings for CPUs with P and E cores
	maxThreads := runtime.NumCPU()
	runtime.GOMAXPROCS(maxThreads)
	if opts.Threads > 0 {
		maxThreads = opts.Threads
	}

	сtxSize := uint32(512)
	if opts.Context > 0 {
		сtxSize = opts.Context
	}

	predictCount := uint32(128)
	if opts.Predict > 0 {
		predictCount = opts.Predict
	}

	temp := float32(0.8)
	if opts.Temp > 0 {
		temp = opts.Temp
	}

	if !opts.Silent {
		showLogo()
	}

	if opts.Prompt == "" || opts.Model == "" {
		Colorize("\n[magenta][ ERROR ][white] Please specify correct model path and prompt with [light_magenta]--model[white] and [light_magenta]--prompt[white] parameters\n\n")
		os.Exit(0)
	}

	params := ModelParams{
		model:       opts.Model,
		interactive: opts.Chat,

		ctxSize:      сtxSize,
		seed:         -1,
		threadsCount: maxThreads,
		predictCount: predictCount,
		repeatLastN:  64,
		partsCount:   -1,
		batchSize:    8,

		topK:          40,
		topP:          0.95,
		temp:          temp,
		repeatPenalty: 1.10,

		memoryFP16: true,
	}

	// --- load the model

	ctx, err := llama.LoadModel(params.model, opts.Silent) // FIXME parts count
	if err != nil {
		Colorize("\n[magenta][ ERROR ][white] Failed to load model [light_magenta]\"%s\"\n\n", params.model)
		os.Exit(0)
	}

	// Add a space in front of the first character to match OG llama tokenizer behavior
	prompt = " " + prompt
	// tokenize the prompt
	embdInp := ml.Tokenize(ctx.Vocab, prompt, true)
	tokenNewline := ml.Tokenize(ctx.Vocab, "\n", false)[0]

	final := "" // model output

	////params.n_keep    = std::min(params.n_keep,    (int) embd_inp.size());
	////params.n_predict = std::min(params.n_predict, n_ctx - (int) embd_inp.size());

	// prefix & suffix for instruct mode
	////inpPrefix := ml.Tokenize(vocab, "\n\n### Instruction:\n\n", true)
	////inpSuffix := ml.Tokenize(vocab, "\n\n### Response:\n\n", false)

	// in instruct mode, we inject a prefix and a suffix to each input by the user
	////if params.instruct {
	////params.interactive = true
	////params.antiprompt.push_back("### Instruction:\n\n");
	////}

	if ml.DEBUG {
		fmt.Printf("\n\n=== TOKENIZER ===\n\n%+v", embdInp)
		for i := 0; i < len(embdInp); i++ {
			fmt.Printf("%d:'%s'  ", embdInp[i], ml.Token2Str(ctx.Vocab, embdInp[i]))
		}
	}

	var embd []uint32

	// TODO: replace with ring-buffer
	lastNTokens := make([]uint32, params.ctxSize, params.ctxSize) // FIXME LEN vs CAP

	////if (params.interactive) {
	////fmt.Printf("== Running in interactive mode. ==\n"
	////#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
	////              " - Press Ctrl+C to interject at any time.\n"
	////#endif
	////              " - Press Return to return control to LLaMa.\n"
	////              " - If you want to submit another line, end your input in '\\'.\n");
	////is_interacting = params.interactive_start || params.instruct;
	////   }

	inputNoEcho := false
	pastCount := uint32(0)
	remainCount := params.predictCount
	consumedCount := uint32(0)

	for remainCount != 0 || params.interactive {

		// --- predict

		if len(embd) > 0 {

			// infinite text generation via context swapping
			// if we run out of context:
			// - take the n_keep first tokens from the original prompt (via n_past)
			// - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch

			if pastCount+uint32(len(embd)) > params.ctxSize {
				leftCount := pastCount - params.keepCount
				pastCount = params.keepCount

				// insert n_left/2 tokens at the start of embd from last_n_tokens
				////embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());
				embd = append(lastNTokens[:leftCount/2], embd...)
			}

			if err := llama.Eval(ctx, embd, uint32(len(embd)), pastCount, params.threadsCount); err != nil {
				fmt.Printf("\n[ERROR] Failed to eval")
				os.Exit(1)
			}
		}

		pastCount += uint32(len(embd))
		embd = []uint32{}

		if len(embdInp) <= int(consumedCount) { // && !isInteracting {

			if params.ignoreEOS {
				ctx.Logits[ml.TOKEN_EOS] = 0
			}

			id := llama.SampleTopPTopK(ctx,
				lastNTokens[params.ctxSize-params.repeatLastN:], params.repeatLastN,
				params.topK, params.topP, params.temp, params.repeatPenalty)

			lastNTokens = lastNTokens[1:] ////last_n_tokens.erase(last_n_tokens.begin());
			lastNTokens = append(lastNTokens, id)

			// replace end of text token with newline token when in interactive mode
			if id == ml.TOKEN_EOS && params.interactive && !params.instruct {
				id = tokenNewline
				////if params.antiprompt.size() != 0) {
				// tokenize and inject first reverse prompt
				////const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false);
				////embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
				////}

			}

			// add it to the context
			embd = append(embd, id)

			// echo this to console
			inputNoEcho = false

			// decrement remaining sampling budget
			remainCount--

		} else {

			// some user input remains from prompt or interaction, forward it to processing
			for len(embdInp) > int(consumedCount) {
				embd = append(embd, embdInp[consumedCount])
				////lastNTokens.erase(last_n_tokens.begin())
				if len(lastNTokens) > 0 { // FIXME GOTZ
					lastNTokens = lastNTokens[1:]
				}
				lastNTokens = append(lastNTokens, embdInp[consumedCount])
				consumedCount++
				if len(embd) >= int(params.batchSize) {
					break
				}
			}

		}

		// --- display text

		if !inputNoEcho {
			for _, id := range embd {

				token := ml.Token2Str(ctx.Vocab, id)
				final += token

				if len(strings.TrimSpace(final)) < len(strings.TrimSpace(prompt)) {
					continue
				}

				out := strings.Split(final, prompt)

				if len(out) == 2 && token == "\n" {
					continue
				}

				if len(strings.TrimSpace(final)) == len(strings.TrimSpace(prompt)) && (token != "\n") && (len(out) == 2) {
					Colorize("\n\n[magenta]▒▒▒ [light_yellow]" + strings.TrimSpace(prompt) + "\n[light_blue]▒▒▒ ")
					continue
				}

				Colorize("[white]" + token)

			}

		}

		////if (n_past > 0 && is_interacting) {
		// potentially set color to indicate we are taking user input
		////set_console_state(CONSOLE_STATE_USER_INPUT);

		////if (params.instruct) {
		////input_consumed = embd_inp.size();
		////embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());

		////printf("\n> ");
		////}

		////std::string buffer;
		////if (!params.input_prefix.empty()) {
		////buffer += params.input_prefix;
		////printf(buffer.c_str());
		/////}

		////std::string line;
		////bool another_line = true;
		////do {
		////if (!std::getline(std::cin, line)) {
		// input stream is bad or EOF received
		////return 0;
		////}
		////if (line.empty() || line.back() != '\\') {
		////another_line = false;
		////} else {
		////line.pop_back(); // Remove the continue character
		////}
		////buffer += line + '\n'; // Append the line to the result
		////} while (another_line);

		// done taking input, reset color
		////set_console_state(CONSOLE_STATE_DEFAULT);

		////auto line_inp = ::llama_tokenize(ctx, buffer, false);
		////embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

		////if (params.instruct) {
		////embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
		////}

		////            remaining_tokens -= line_inp.size();

		////            input_noecho = true; // do not echo this again
		////        }

		////if (n_past > 0) {
		////        is_interacting = false;

		////    }
		////}

		// end of text token
		////if (embd.back() == llama_token_eos()) {
		////if (params.instruct) {
		////is_interacting = true;
		////} else {
		////fprintf(stderr, " [end of text]\n");
		////break;
		////}
		////}

		// In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
		////if (params.interactive && n_remain <= 0 && params.n_predict != -1) {
		////remaining_tokens = params.n_predict;
		////is_interacting = true;
		////}
	}
}

// Join colorstring and go-colorable to allow colors both on Mac and Windows
// TODO Implement as a small library
func Colorize(format string, opts ...interface{}) (n int, err error) {
	var DefaultOutput = colorable.NewColorableStdout()
	return fmt.Fprintf(DefaultOutput, colorstring.Color(format), opts...)
}

// TODO Show actual version
func showLogo() {

	// https://patorjk.com/software/taag/#p=display&f=3-D&t=llama.go%0A%0ALLaMA.go
	// Isometric 1, Modular, Rectangles, Rozzo, Small Isometric 1, 3-D

	logo := `                                                    
  /88       /88         /888/888   /88/8888/88   /888/888      /8888/88   /888/888    
  /888      /888      /888/ /888 /888/8888/888 /888/ /888     /8888 //   /8888//888  
  /8888/88  /8888/88  /8888/8888 /888/8888/888 /8888/8888 /88 /8888/8888 /888 /8888 
  /8888/888 /8888/888 /888 /8888 /888//88 /888 /888 /8888 /888//8888/88  //888/888
  //// ///  //// ///  ///  ////  ///  //  ///  ///  ////  ///  //// //    /// ///`

	logoColored := ""
	prevColor := ""
	color := ""
	line := 0
	colors := []string{"[black]", "[light_blue]", "[magenta]", "[light_magenta]", "[light_blue]"}

	for _, char := range logo {
		if char == '\n' {
			line++
		} else if char == '/' {
			color = "[blue]"
		} else if char == '8' {
			color = colors[line]
			char = '▒'
		}
		if color == prevColor {
			logoColored += string(char)
		} else {
			logoColored += color + string(char)
		}
	}

	Colorize(logoColored) //colorstring.Printf(logoColored)
	Colorize("\n\n   [magenta]▒▒▒▒[light_magenta] [ LLaMA.go v0.8 ] [light_blue][ Pure Go implementation of Meta's LLaMA GPT model ] [magenta]▒▒▒▒\n\n")
}

/*
void perplexity(llama_context * ctx, const gpt_params & params) {
    // Download: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research
    // Run `./main --perplexity -m models/7B/ggml-model-q4_0.bin -f wiki.test.raw`
    // Output: `perplexity: 13.5106 [114/114]`
    auto tokens = ::llama_tokenize(ctx, params.prompt, true);

    int count = 0;
    double nll = 0.0;
    int seq_count = tokens.size() / params.n_ctx;

    fprintf(stderr, "%s : calculating perplexity over %d chunks\n", __func__, seq_count);

    for (int i = 0; i < seq_count; ++i) {
        int start = i * params.n_ctx;
        int end = start + params.n_ctx - 1;
        std::vector<llama_token> embd(tokens.begin() + start, tokens.begin() + end);
        auto start_t = std::chrono::high_resolution_clock::now();
        if (llama_eval(ctx, embd.data(), embd.size(), 0, params.n_threads)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return;
        }
        auto end_t = std::chrono::high_resolution_clock::now();
        if (i == 0) {
            double seconds = std::chrono::duration<double>(end_t - start_t).count();
            printf("%.2f seconds per pass - ETA %.2f hours\n", seconds, (seconds * seq_count) / (60.0*60.0));
        }
        // We get the logits for all the tokens in the context window (params.n_ctx)
        // from llama_eval above.  Now, based on https://huggingface.co/docs/transformers/perplexity,
        // calculate the perplexity over the last half the window (so the model always has
        // some context to predict the token).
        //
        // We rely on the fact that attention in the forward pass only looks at previous
        // tokens here, so the logits returned for each token are an accurate representation
        // of what the model would have predicted at that point.
        //
        // Example, we have a context window of 512, we will compute perplexity for each of the
        // last 256 tokens.  Then, we split the input up into context window size chunks to
        // process the entire prompt.

        auto logits = llama_get_logits(ctx);
        for (int j = params.n_ctx / 2; j < params.n_ctx - 1; ++j) {
            // Calculate probability of next token, given the previous ones.
            int n_vocab = llama_n_vocab(ctx);
            std::vector<float> tok_logits(
                logits + j * n_vocab,
                logits + (j + 1) * n_vocab);
            double prob = softmax(tok_logits)[tokens[start + j + 1]];
            nll += -std::log(prob);
            ++count;
        }
        // perplexity is e^(average negative log-likelihood)
        printf("[%d]%.4lf,", i + 1, std::exp(nll / count));
        fflush(stdout);
    }
    printf("\n");
}*/
