"""Sample answers from LLMs on QA task."""
import gc
import os
import logging
import random
from tqdm import tqdm

import numpy as np
import torch
from .uncertainty.utils import utils
from .compute_uncertainty_measures import main as main_compute


utils.setup_logger()


def main(args):
    path = '/mnt/data/liusiyuan/prompt_ood/cache_data'
    experiment_details = {'args': args}
    random.seed(args.random_seed)
    if not os.path.exists(path):
        os.makedirs(path)

    # Get indices of answerable and unanswerable questions and construct prompt.
    prompt_indices = args.prompt
    experiment_details['prompt_indices'] = prompt_indices

    # Create Few-Shot prompt.
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(prompt_indices, BRIEF, arg, make_prompt)
    experiment_details['prompt'] = prompt
    experiment_details['BRIEF'] = BRIEF
    logging.info('Prompt is: %s', prompt)

    # Initialize model.
    model = utils.init_model(args)

    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
    logging.info(80 * 'x')

    # This will store all input data and model predictions.
    generations, results_dict= {}, {}
    question = args.prompt
    generations[args.id] = {'question': question}
    current_input = make_prompt(
        question, BRIEF, args.brief_always and args.enable_brief)
    local_prompt = prompt + current_input

    logging.info('Current input: '.ljust(15) + current_input)

    full_responses = []

    # We sample one low temperature answer on which we will compute the
    # accuracy and args.num_generation high temperature answers which will
    # be used to estimate the entropy variants.


    # args.num_generations default = 10
    num_generations = args.num_generations + 1

    for i in range(num_generations):

        # Temperature for first generation is always `0.1`.
        temperature = args.temperature

        predicted_answer, token_log_likelihoods, embedding = model.predict(
            local_prompt, temperature)
        embedding = embedding.cpu() if embedding is not None else None

        logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
        # Aggregate predictions over num_generations.
        full_responses.append(
            (predicted_answer, token_log_likelihoods, embedding))

    # Append all predictions for this example to `generations`.
    generations[args.id]['responses'] = full_responses


    # Save generations for that split.
    dataset_split = 'train'
    utils.save(generations, f'{dataset_split}_generations.pkl', path)
    utils.save(results_dict, 'uncertainty_measures.pkl', path)
    utils.save(experiment_details, 'experiment_details.pkl', path)
    logging.info('Run complete.')
    del model


if __name__ == '__main__':

    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    # First sample generations from LLM.
    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')

    if args.compute_uncertainties:
        # Follow with uncertainty calculation script by default.
        args.assign_new_wandb_id = False
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        main_compute(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')
