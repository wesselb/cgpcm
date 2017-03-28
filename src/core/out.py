from termcolor import colored
import sys

import util

level = 0
width = 25
indenter = '    '


def _out(msg):
    sys.stdout.write(msg)


def state(msg):
    """
    State a message.

    :param msg: messsage
    """
    _out(indenter * level + '{}\n'.format(msg))


def error(msg):
    """
    State an error message.
    
    :param msg: error message
    """
    _out(colored('ERROR: {}\n'.format(msg), 'red'))


def section(name):
    """
    Start a section.

    :param name: name of section
    """
    state(name)
    indent()


def section_end():
    """
    End a section.
    """
    dedent()


def kv(key, value, mod='', unit=''):
    """
    Show a key-value pair.

    :param key: key
    :param value: value
    :param mod: print modifier of value
    :param unit: unit of value
    """
    _out(indenter * level)
    form = '{{key:{width:d}s}}{{value:{mod}}} {{unit}}\n'.format(width=width,
                                                                 mod=mod)
    _out(form.format(key=key + ':', value=value, unit=unit))


def indent():
    """
    Indent one level.
    """
    global level, width
    level += 1
    width += len(indenter)


def dedent():
    """
    Dedent one level.
    """
    global level, width
    level -= 1
    width -= len(indenter)


def eat(num=1):
    """
    Erase previously printed lines.

    :param num: number of lines to erase
    """
    control_up_line = '\x1b[1A'
    control_erase_line = '\x1b[2K'
    for i in range(num):
        sys.stdout.write(control_erase_line + control_up_line)


def dict_(d, numeric_mod=''):
    """
    Show a dictionary.

    :param d: dictionary
    :param numeric_mod: print modifier of numeric values
    """
    for k, v in sorted(d.items(), key=lambda x: x[0]):
        if type(v) == dict:
            section(k)
            dict_(v, numeric_mod=numeric_mod)
            section_end()
        else:
            if util.is_numeric(v):
                kv(k, v, mod=numeric_mod)
            else:
                kv(k, v)
