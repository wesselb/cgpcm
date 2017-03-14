#!/usr/bin/env python
import importlib
import pickle
import argparse

from core.tfutil import *
import core.util as util
import core.out as out
import core.exp as exp

parser = argparse.ArgumentParser(description='Run a task.')
parser.add_argument('-t', '--task', type=str, action='append', nargs='*',
                    help='task to run with options', default=[])
parser.add_argument('-d', '--debug', nargs='*', help='debug options',
                    default=[])

args = parser.parse_args()
out.kv('number of tasks', len(args.task))

for i, task_args in enumerate(args.task):
    out.section('task')
    out.kv('task number', '{}/{}'.format(i + 1, len(args.task)))
    out.kv('specification', ' '.join(task_args))

    # Fetch task
    module = importlib.import_module('tasks.{}'.format(task_args[0]))
    task = module.Experiment(options=task_args[1:])

    # Initialise
    out.kv('name', task.config.name)
    tf.reset_default_graph()
    with Session() as sess:
        util.seed(task.config.seed)

        # Train
        out.section('training')
        exp.train(sess, task, args.debug)
        out.section_end()

        # Save resulting task
        out.section('saving')
        task.make_pickleable(sess)
        fn = 'tasks/cache/{}.pickle'.format(task.config.fn)
        util.mkdirs(fn)
        with open(fn, 'w') as f:
            pickle.dump(task, f)
        out.section_end()

    out.section_end()
