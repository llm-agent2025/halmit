from generate_tools import LLMCALL
import utils
import pandas as pd
import numpy as np
import logging
logging.basicConfig(filename='Treatment.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import random
import os
import io
import pickle
from collections import defaultdict
import json
import torch
from semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta




def generate_answer(question, model_name, max_new_tokens, temperature):
    parser = utils.get_parser()
    args, _ = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)
    random.seed(args.random_seed)


    # Create Few-Shot prompt.
    brief = "Answer the following question about 'Treatment' as briefly as possible.\n"
    prompt = brief + question
    # prompt = question
    logging.info('Prompt is: %s', question)

    # Initialize model.
    model = LLMCALL(model_name = model_name, stop_sequences = None, max_new_tokens = max_new_tokens)

    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
    logging.info(80 * 'x')
    answer, confidence = model.hf_gen(
        question = prompt)
    # logging.info('Answer is: '.ljust(15) + ' : ' + answer)

    # Start Sample
    # This will store all input data and model predictions.
    validation_generations = {}
    validation_generations[args.id] = {'question': question}

    full_responses = []

    # We sample one low temperature answer on which we will compute the
    # accuracy and args.num_generation high temperature answers which will
    # be used to estimate the entropy variants.

    # args.num_generations default = 10
    num_generations = args.num_generations + 1

    for i in range(num_generations):

        predicted_answer, token_log_likelihoods, embedding = model.predict(
            question = prompt, temperature = temperature)
        max_attempts = 3 
        attempts = 0  
        while predicted_answer == 1:
            predicted_answer, token_log_likelihoods, embedding = model.predict(
            question = prompt, temperature = temperature)
            attempts += 1

            if attempts >= max_attempts:
                model.reload()
                del model
                logging.info('delet model')
                max_new_tokens = max_new_tokens + 500
                model = LLMCALL(model_name = model_name, stop_sequences = None, max_new_tokens = max_new_tokens) 
                attempts = 0 

        embedding = embedding.cpu() if embedding is not None else None
        logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
        # Aggregate predictions over num_generations.
        full_responses.append(
            (predicted_answer, token_log_likelihoods, embedding))

    # Append all predictions for this example to `generations`.
    validation_generations[args.id]['responses'] = full_responses

    logging.info('Sample complete.')
    torch.cuda.empty_cache()
    # logging.info('Beginning loading for entailment model.')
    if args.entailment_model == 'deberta':
        entailment_model = EntailmentDeberta()
        # logging.info('Entailment model loading complete.')


    entropies = defaultdict(list)

    # Loop over datapoints and compute validation embeddings and entropies.
    for idx, tid in enumerate(validation_generations):
        example = validation_generations[tid]
        question = example['question']
        full_responses = example["responses"]

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        if args.compute_predictive_entropy:
            # Token log likelihoods. Shape = (n_sample, n_tokens)
            if not args.use_all_generations:
                log_liks = [r[1] for r in full_responses[:args.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            for i in log_liks:
                assert i

            if args.condition_on_question and args.entailment_model == 'deberta':
                responses = [f'{question} {r}' for r in responses]

            # Compute semantic ids.
            semantic_ids = get_semantic_ids(
                responses, model=entailment_model,
                strict_entailment=args.strict_entailment, example=example)

            # result_dict['semantic_ids'].append(semantic_ids)

            # Length normalization of generation probabilities.
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

            # Compute semantic entropy.
            log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
            pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            entropies['semantic_entropy'].append(pe)

            # pylint: disable=invalid-name
            log_str = 'semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s'
            entropies_fmt = ', '.join([f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])
            # pylint: enable=invalid-name
            logging.info(80*'#')
            logging.info('High Temp Generation:')
            logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)
      

    if args.compute_predictive_entropy:
        entailment_model.save_prediction_cache()
    return answer, pe

    

def process_csv(input_csv, output_csv, topic):
    df = pd.read_csv(input_csv)
    df['answer'] = None
    df['semantic_entropy'] = None
    df['judge'] = None
    for index, row in df.iterrows():
        question = row['prompt']
        if pd.isna(question): 
            continue
        try:

            answer, semantic_entropy = generate_answer(question = question, model_name = 'mistral7b', max_new_tokens = 1000, temperature = 1)

            logging.info(80 * '=')
            judge_prompt = utils.construct_action_prompt(action = 'judge', question = question, topic = topic)
            judge = LLMCALL(model_name = 'gpt', stop_sequences = None, max_new_tokens = 100).gpt_api(question = judge_prompt)
            logging.info('Judge: %s', judge)
            logging.info(80 * '=')
            df.at[index, 'answer'] = answer
            df.at[index, 'semantic_entropy'] = semantic_entropy
            df['judge'] = judge
        except Exception as e:
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    input_csv = 'halmit/Treatment.csv'     
    output_csv = 'halmit/Treatment_results.csv' 
    topic = 'Treatment'
    process_csv(input_csv, output_csv, topic)





