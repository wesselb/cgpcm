#!/usr/bin/env python
import importlib
import pickle
import argparse

import core.out as out
import core.exp as exp

parser = argparse.ArgumentParser(description='Run a task.')
parser.add_argument('plot', type=str, nargs='+',
                    help='type of plot', default=[])
parser.add_argument('-t', '--task', type=str, action='append', nargs='*',
                    help='task to plot with options', default=[])
parser.add_argument('-d', '--debug', nargs='*', help='debug options',
                    default=[])
parser.add_argument('-s', '--show', action='store_true', help='show plot')

args = parser.parse_args()
out.kv('plot type', args.plot[0])
out.kv('plot options', ' '.join(args.plot[1:]))

out.section('fetching tasks')
tasks = []
for task_args in args.task:
    out.kv('specification', ' '.join(task_args))

    # Fetch task
    module = importlib.import_module('tasks.{}'.format(task_args[0]))
    fn = module.Experiment(options=task_args[1:]).config.fn
    with open('tasks/cache/{}.pickle'.format(fn)) as f:
        tasks.append(pickle.load(f))
out.section_end()

out.section('plotting')
p, fn = exp.plot(args.plot[0], tasks, options=args.plot[1:])
p.save('output/out_{}_{}.pdf'.format(args.plot[0], fn))
if args.show:
    p.show()
out.section_end()
