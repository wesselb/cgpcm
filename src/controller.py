#!/usr/bin/env python
import importlib
import pickle
import argparse
import matplotlib.pyplot as plt
import sys

from core.tf_util import *
from core.options import Options
import core.util as util
import core.out as out
import core.experiment as exp


# Compatibility
sys.modules['core.exp'] = exp


def start_session():
    """
    Reset default graph and start a TensorFlow session.
    :return: TensorFlow session
    """
    tf.reset_default_graph()
    return Session()


def fetch(specifications, remote):
    """
    Fetch tasks from their specifications.

    :param specifications: specifications
    :param remote: use remote cache
    :return: tasks
    """
    out.section('fetching tasks')
    out.kv('from remote', 'yes' if remote else 'no')
    fetched_tasks = []
    for specification in specifications:
        task_name, task_options = specification[0], specification[1:]
        out.kv('specification', ' '.join(specification))

        # Fetch task
        try:
            module = importlib.import_module('tasks.{}'.format(task_name))
            fp = module.Experiment(options=task_options).config.fp
            path = 'tasks/{}cache/'.format('remote/' if remote else '') \
                   + str(fp) + '.pickle'
            with open(path) as f:
                fetched_tasks.append(pickle.load(f))
        except IOError:
            out.error('couldn\'t fetch task: cache not found')
            fetched_tasks.append(None)
        except ImportError:
            out.error('couldn\'t fetch task: task not found')
            fetched_tasks.append(None)

    out.section_end()
    return fetched_tasks


def plot(plot_specification, tasks):
    """
    Plot a list of tasks.

    :param plot_specification: specification of plot
    :param tasks: tasks
    """
    plot_type, plot_options = plot_specification[0], plot_specification[1:]
    out.kv('plot type', plot_type)
    out.section('plotting')
    p, fp = exp.plot(plot_type, tasks, args=plot_options)
    path = 'output/' + str(fp) + '.pdf'
    util.mkdirs(path)
    p.save(path)
    out.section_end()


def train(task_specification, debug_options):
    """
    Generate and train a task.

    :param task_specification: task specification
    :param debug_options: debug options
    :return: TensorFlow session, task, and model
    """
    task_name, task_options = task_specification[0], task_specification[1:]
    module = importlib.import_module('tasks.{}'.format(task_name))
    task = module.Experiment(options=task_options)

    # Start TensorFlow session and seed
    sess = start_session()
    util.seed(task.config.seed)

    # Train
    out.section('training')
    mod = exp.train(sess, task, debug_options)
    out.section_end()

    return sess, task, mod


def load_trained_model(sess, task, debug_options):
    """
    Load a trained model from a task.
    
    :param sess: TensorFlow session
    :param task: task
    :param debug_options: debug options
    :return: trained model
    """
    out.section('loading trained model')
    mod = exp.mod_from_task(sess, task, debug_options)
    out.section_end()
    return mod


def predict(sess, task, mod, debug_options):
    """
    Predict a task.
    
    :param sess: TensorFlow session
    :param task: task
    :param mod: trained model
    :param debug_options: debug options
    :return: 
    """
    out.section('predicting')
    exp.predict(sess, task, mod, debug_options)
    out.section_end()


def save(sess, task, debug_options):
    """
    Save a task.
    
    :param sess: TensorFlow session
    :param task: tasks
    :param debug_options: debug options
    """
    out.section('saving')
    task.make_pickleable(sess)
    path = 'tasks/cache/' + str(task.config.fp) + '.pickle'
    util.mkdirs(path)
    with open(path, 'w') as f:
        pickle.dump(task, f)
    out.section_end()


def report(task, debug_options):
    """
    Generate report for a task.
    
    :param task: task
    :param debug_options: debug_options
    """
    out.section('reporting task')
    report = task.report()
    out.dict_(report, numeric_mod='.3e')
    path = 'output/stats/' + str(task.config.fp) + '.pickle'
    util.mkdirs(path)
    with open(path, 'w') as f:
        pickle.dump(report, f)
    out.section_end()


def process_task_specific_argument(arg, num_tasks):
    """
    Process a task-specific argument.
    
    :param num_tasks: total number of tasks
    :param arg: argument
    :return: processed argument
    """
    if type(arg) == int:
        # If only one is specified, then wrap it appropriately
        return [arg]
    elif arg == []:
        # If none are specified, then specify all
        return range(num_tasks)
    elif arg == None:
        # If the argument is not specified, then none are specified
        return []
    return arg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a task.')
    parser.add_argument('-t', '--task', type=str, action='append', nargs='+',
                        help='task to run with options', default=[])
    parser.add_argument('-p', '--plot', type=str, action='append', nargs='+',
                        help='plot with options', default=[])

    parser.add_argument('--debug', nargs='*', help='debug options', default=[])
    parser.add_argument('--train', type=int, action='store', nargs='*',
                        help='train all tasks, or specify indices of tasks to '
                             'train; a task to train will also be predicted')
    parser.add_argument('--predict', type=int, action='store', nargs='*',
                        help='predict all tasks, or specify indices of tasks '
                             'to predict')
    parser.add_argument('--show', action='store_true', help='show plots')
    parser.add_argument('--report', type=int, action='store', nargs='*',
                        help='report all tasks, or specify indices of tasks '
                             'to report')
    parser.add_argument('--remote', action='store_true',
                        help='use remote task cache')

    args = parser.parse_args()

    num_tasks = len(args.task)
    num_plots = len(args.plot)

    debug_options = Options('debug options')
    debug_options.add_option('kernel', 'plot kernel and filter and then exit')
    debug_options.add_option('fix-alpha', 'do not learn alpha')
    debug_options.parse(args.debug)

    # Process task-specific arguments
    args.train = process_task_specific_argument(args.train, num_tasks)
    args.predict = process_task_specific_argument(args.predict, num_tasks)
    args.report = process_task_specific_argument(args.report, num_tasks)

    # Furthermore, any task to be trained will also be predicted
    args.predict = sorted(set(args.train) | set(args.predict))

    out.kv('number of tasks', num_tasks)
    out.kv('number of plots', num_plots)
    out.kv('learning tasks',
           ', '.join(map(str, args.train)) if args.train else 'none')
    out.kv('predicting tasks',
           ', '.join(map(str, args.predict)) if args.predict else 'none')

    # Fetch anything that is already available
    tasks = fetch(args.task, remote=args.remote)

    # Learning and prediction of tasks
    for i, task_specification in enumerate(args.task):
        task = tasks[i]
        out.section('processing task')
        out.kv('task index', '{}/{}'.format(i, num_tasks - 1))
        out.kv('specification', ' '.join(task_specification))

        if i in args.train:
            # Train and predict
            sess, task, mod = train(task_specification, debug_options)
            predict(sess, task, mod, debug_options)
            save(sess, task, debug_options)
            tasks[i] = task  # Update fetched tasks
        elif i in args.predict:
            # Just predict, assuming that a trained model is available
            sess = start_session()
            mod = load_trained_model(sess, task, debug_options)
            task.mod = mod
            predict(sess, task, mod, debug_options)
            save(sess, task, debug_options)

        if i in args.report:
            report(task, debug_options)

        out.section_end()

    # Generate plots
    for i, plot_specification in enumerate(args.plot):
        out.section('processing plot')
        out.kv('plot index', '{}/{}'.format(i, num_plots - 1))
        out.kv('specification', ' '.join(plot_specification))

        plot(plot_specification, tasks)

        out.section_end()

    # Show plots
    if args.plot and args.show:
        out.section('showing plots')
        plt.show()
        out.section_end()
