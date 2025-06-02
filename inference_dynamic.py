import torch
import torch.nn as nn
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer, Timer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2BaseGenerator, ExLlamaV2Sampler

def evaluate_ppl(model, cache, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_dataset_text = "\n\n".join(test_dataset["text"])

    ids = tokenizer.encode(test_dataset_text)
    if isinstance(ids, list):
        ids = torch.tensor(ids, dtype=torch.long).to(device)

    if ids.dim() == 1:
        input_ids = ids.unsqueeze(0)
    elif ids.dim() == 2:
        input_ids = ids
    else:
        raise ValueError(f"Unexpected ids.dim() = {ids.dim()}")

    seq_len = 2048
    nsamples = input_ids.numel() // seq_len
    nlls = []

    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = input_ids[:, (i * seq_len):((i + 1) * seq_len)]

        with torch.no_grad():
            cache.current_seq_len = 0
            lm_logits = model.forward(batch, cache)

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].to(device)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss * seq_len
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seq_len))
    return ppl.item()


def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)

    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'

    ### === TODO: Load your model (you may change this part) ===

    #####################################
    ### === Load ExLlamaV2 model === ###

    model_dir = "models/EAI_Final_model"

    # Load config
    config = ExLlamaV2Config(model_dir)
    config.arch_compat_overrides()

    # Load model and cache
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, max_seq_len=4096)
    model.load_autosplit(cache, progress = True)

    # Load tokenizer
    tokenizer = ExLlamaV2Tokenizer(config)

    # Initialize generator
    generator = ExLlamaV2DynamicGenerator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        paged=False,
        filter_background_eval=True,
    )

    gen_settings = ExLlamaV2Sampler.Settings()
    gen_settings.temperature = 0.8
    gen_settings.top_k = 50
    gen_settings.top_p = 0.95
    gen_settings.token_repetition_penalty = 1.1


    #####################################
    ### === Warm Up === ###

    warmup_prompt = "Explain what AI is."
    token_ids = tokenizer.encode(warmup_prompt)

    for i in tqdm(range(5), desc="Warm Up..."):
        
        generated = generator.generate(
            prompt=warmup_prompt,
            max_new_tokens=max_new_tokens,
            gen_settings=gen_settings,
            add_bos=True,
        )

    ####################################################################
    ### === Test Inference === ###

    prompt = "How to learn a new language?"

    token_ids = tokenizer.encode(prompt)
    len_prompt = token_ids.shape[1]

    input_ids = token_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    tputs = []
    time_record = []
    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        generated = generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            gen_settings=gen_settings,
            add_bos=True,
        )

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)

        token_gen = tokenizer.encode(generated)
        len_gen = token_gen.shape[1]

        num_generated_tokens = len_gen - len_prompt
        tput = num_generated_tokens / (elapsed_ms / 1000)

        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)

    response = generated
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\n')
    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')

    ####################################################################

    ### Your final throughput result ###
    print(f'Throughput: {org_tput} toks/s')

    ppl = evaluate_ppl(model, cache, tokenizer, device)
    print(f"Perplexity (PPL): {ppl}")

    # # Save results to CSV
    import csv
    rounded_tput = round(org_tput, 1)
    ppl = round(ppl, 2)

    with open("result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, ppl])
        writer.writerow([1, rounded_tput])

if __name__ == '__main__':
    main()