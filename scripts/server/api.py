#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive interface to full DrQA pipeline."""

import torch
import argparse
import code
import prettytable
import logging
import time

from termcolor import colored
from drqa import pipeline
from drqa.retriever import utils

# Server code
from flask import Flask
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify

app = Flask(__name__)
api = Api(app)

class Question(Resource):
    def get(self, question_string):
        query = ' '.join(question_string.split('%20'))
        if query[len(query)-1] != '?': query += '?'
        process_result = process(query, None, 5)
        result = {'query': query,'results': [dict(zip(["result_number","doc_id","span","span_score","doc_score","context"],[i,p["doc_id"],p["span"],p["span_score"],p["doc_score"],p['context']['text']])) for i,p in enumerate(process_result) ]}
        return jsonify(result)

api.add_resource(Question, '/question/<question_string>')

# -----------------------------------------

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--reader-model', type=str, default=None,
                    help='Path to trained Document Reader model')
parser.add_argument('--retriever-model', type=str, default=None,
                    help='Path to Document Retriever model (tfidf)')
parser.add_argument('--doc-db', type=str, default=None,
                    help='Path to Document DB')
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to "
                          "use (e.g. 'corenlp')"))
parser.add_argument('--candidate-file', type=str, default=None,
                    help=("List of candidates to restrict predictions to, "
                          "one candidate per line"))
parser.add_argument('--no-cuda', action='store_true',
                    help="Use CPU only")
parser.add_argument('--gpu', type=int, default=-1,
                    help="Specify GPU device id to use")
parser.add_argument('--num_workers', type=int, default=None,
                    help="Specify number of CPU workers")
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

if args.candidate_file:
    logger.info('Loading candidates from %s' % args.candidate_file)
    candidates = set()
    with open(args.candidate_file) as f:
        for line in f:
            line = utils.normalize(line.strip()).lower()
            candidates.add(line)
    logger.info('Loaded %d candidates.' % len(candidates))
else:
    candidates = None

logger.info('Initializing pipeline...')
DrQA = pipeline.DrQA(
    cuda=args.cuda,
    fixed_candidates=candidates,
    reader_model=args.reader_model,
    ranker_config={'options': {'tfidf_path': args.retriever_model}},
    db_config={'options': {'db_path': args.doc_db}},
    tokenizer=args.tokenizer,
    num_workers = args.num_workers
)


# ------------------------------------------------------------------------------
# Processing
# ------------------------------------------------------------------------------

def process(question, candidates=None, top_n=1, n_docs=5):
    t0 = time.time()
    predictions = DrQA.process(
        question, candidates, top_n, n_docs, return_context=True
    )
    table = prettytable.PrettyTable(
        ['Rank', 'Answer', 'Doc', 'Answer Score', 'Doc Score']
    )
    for i, p in enumerate(predictions, 1):
        table.add_row([i, p['span'], p['doc_id'],
                       '%.5g' % p['span_score'],
                       '%.5g' % p['doc_score']])
    print('Top Predictions:')
    print(table)
    print('\nContexts:')
    for p in predictions:
        text = p['context']['text']
        start = p['context']['start']
        end = p['context']['end']
        output = (text[:start] +
                  colored(text[start: end], 'green', attrs=['bold']) +
                  text[end:])
        print('[ Doc = %s ]' % p['doc_id'])
        print(output + '\n')
    print('Time: %.4f' % (time.time() - t0))
    return predictions

if __name__ == '__main__':
    app.run(debug=True)
