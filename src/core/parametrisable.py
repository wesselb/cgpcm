class Parametrisable(object):
    """
    Class whose keywords given to the constructor become attributes.

    Required parameters can be specified through overriding `_required_pars`.
    """
    _required_pars = []

    def __init__(self, **pars):
        # Verify that all required parameters are specified
        if not all(k in pars for k in self._required_pars):
            unspecified_pars = [k for k in self._required_pars
                                if k not in pars]
            formatted = ', '.join('"{}"'.format(k) for k in unspecified_pars)
            raise RuntimeError('must specify {}'.format(formatted))

        # Set parameters
        for k, v in pars.items():
            setattr(self, k, v)

        self._pars = pars
