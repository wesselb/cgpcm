import sys

level = 0
total_width = 25
width = 25
indenter = '  '
first_output = True


def _out(msg):
    global first_output
    first_output = False
    sys.stdout.write(msg)


def state(msg):
    """
    State a message.

    :param msg: messsage
    """
    _out(indenter * level + '{{:{}}}\n'.format(width).format(msg))


def section(name):
    """
    Start a section.

    :param name: name of section
    """
    if level == 0 and not first_output:
        state('')
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
