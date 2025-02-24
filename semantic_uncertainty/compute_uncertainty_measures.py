"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
# import wandb

from .analyze_results import analyze_run
# from .uncertainty.data.data_utils import load_ds
# from .uncertainty.uncertainty_measures.p_ik import get_p_ik
from .uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from .uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from .uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from .uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from .uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from .uncertainty.uncertainty_measures.semantic_entropy import context_entails_response
from .uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from .uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from .uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from .uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo
from .uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
# from .uncertainty.uncertainty_measures import p_true as p_true_utils
from .uncertainty.utils import utils


utils.setup_logger()

EXP_DETAILS = 'experiment_details.pkl'


def main(args):
    path = '/mnt/data/liusiyuan/prompt_ood/cache_data'

    # if args.train_wandb_runid is None:
    #     args.train_wandb_runid = args.eval_wandb_runid

    # # user = os.environ['USER']
    # user = 'lsy'
    # # scratch_dir = os.getenv('SCRATCH_DIR', '.')
    # scratch_dir = 'scratch'
    # # wandb_dir = f'{scratch_dir}/{user}/uncertainty'
    # slurm_jobid = 'test'
    # project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"
    # if args.assign_new_wandb_id:
    #     logging.info('Assign new wandb_id.')
    #     api = wandb.Api()
    #     old_run = api.run(f'{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}')
    #     wandb.init(
    #         entity=args.entity,
    #         project=project,
    #         dir=wandb_dir,
    #         notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
    #         # For convenience, keep any 'generate_answers' configs from old run,
    #         # but overwrite the rest!
    #         # NOTE: This means any special configs affecting this script must be
    #         # called again when calling this script!
    #         config={**old_run.config, **args.__dict__},
    #     )

    #     def restore(filename):
    #         old_run.file(filename).download(
    #             replace=True, exist_ok=False, root=wandb.run.dir)

    #         class Restored:
    #             name = f'{wandb.run.dir}/{filename}'

    #         return Restored
    # else:
    #     logging.info('Reuse active wandb id.')

    #     def restore(filename):
    #         class Restored:
    #             name = f'{wandb.run.dir}/{filename}'
    #         return Restored

    # if args.train_wandb_runid != args.eval_wandb_runid:
    #     logging.info(
    #         "Distribution shift for p_ik. Training on embeddings from run %s but evaluating on run %s",
    #         args.train_wandb_runid, args.eval_wandb_runid)

    #     is_ood_eval = True  # pylint: disable=invalid-name
    #     api = wandb.Api()
    #     old_run_train = api.run(f'{args.restore_entity_train}/semantic_uncertainty/{args.train_wandb_runid}')
    #     filename = 'train_generations.pkl'
    #     old_run_train.file(filename).download(
    #         replace=True, exist_ok=False, root=wandb.run.dir)
    #     with open(f'{wandb.run.dir}/{filename}', "rb") as infile:
    #         train_generations = pickle.load(infile)
    #     wandb.config.update(
    #         {"ood_training_set": old_run_train.config['dataset']}, allow_val_change=True)
    # else:
    #     is_ood_eval = False  # pylint: disable=invalid-name
    #     if args.compute_p_ik or args.compute_p_ik_answerable:
    #         train_generations_pickle = restore('train_generations.pkl')
    #         with open(train_generations_pickle.name, 'rb') as infile:
    #             train_generations = pickle.load(infile)

    # wandb.config.update({"is_ood_eval": is_ood_eval}, allow_val_change=True)

    # Load entailment model.
    if args.compute_predictive_entropy:
        logging.info('Beginning loading for entailment model.')
        if args.entailment_model == 'deberta':
            entailment_model = EntailmentDeberta()
        elif args.entailment_model == 'gpt-4':
            entailment_model = EntailmentGPT4(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-3.5':
            entailment_model = EntailmentGPT35(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-4-turbo':
            entailment_model = EntailmentGPT4Turbo(args.entailment_cache_id, args.entailment_cache_only)
        elif 'llama' in args.entailment_model.lower():
            entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
        else:
            raise ValueError
        logging.info('Entailment model loading complete.')

    # Restore outputs from `generate_answrs.py` run.
    with open(os.path.join(path, 'uncertainty_measures.pkl') , "rb") as infile:
        result_dict = pickle.load(infile)
    result_dict['semantic_ids'] = []

    with open(os.path.join(path, 'train_generations.pkl'), 'rb') as infile:
        validation_generations = pickle.load(infile)

    entropies = defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    p_trues = []
    count = 0  # pylint: disable=invalid-name

    # def is_answerable(generation):
    #     return len(generation['reference']['answers']['text']) > 0

    # Loop over datapoints and compute validation embeddings and entropies.
    for idx, tid in enumerate(validation_generations):

        example = validation_generations[tid]
        question = example['question']
        full_responses = example["responses"]
        # most_likely_answer = example['most_likely_answer']

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]


        # validation_answerable.append(is_answerable(example))
        # validation_embeddings.append(most_likely_answer['embedding'])
        # logging.info('validation_is_true: %f', validation_is_true[-1])

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

            result_dict['semantic_ids'].append(semantic_ids)

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
            logging.info('NEW ITEM %d at id=`%s`.', idx, tid)
            logging.info('Question:')
            logging.info(question)
            # logging.info('Low Temperature Generation:')
            # logging.info(most_likely_answer['response'])
            logging.info('High Temp Generation:')
            logging.info([r[0] for r in full_responses])
            logging.info('High Temp Generation:')
            logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)
      
    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()

    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)

    if args.compute_predictive_entropy:
        entailment_model.save_prediction_cache()

    if args.analyze_run:
        # Follow up with computation of aggregate performance metrics.
        logging.info(50 * '#X')
        logging.info('STARTING `analyze_run`!')
        analyze_run(wandb.run.id)
        logging.info(50 * '#X')
        logging.info('FINISHED `analyze_run`!')

    return entropies['semantic_entropy']


if __name__ == '__main__':
    parser = utils.get_parser(stages=['compute'])
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info("Args: %s", args)

    main(args)
