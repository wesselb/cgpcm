from core.utils import *
import operator

if __name__ == '__main__':
    # For debugging convenience
    sess = tf.Session()


def const(x):
    """
    Constant polynomial.

    :param x: constant
    :return: polynomial
    """
    return Poly(Term(x))


def var(x, power=1):
    """
    Polynomial consisting of just a single variable.

    :param x: variable name
    :param power: power
    :return: polynomial
    """
    return Poly(Term(1., Factor(x, power)))


def kh(alpha, gamma, x, y):
    """
    Kernel of :math:`h` process.

    :param alpha: :math:`\\alpha`
    :param gamma: :math:`\\gamma`
    :param x: first variable
    :param y: second variable
    :return: exponentiated quadratic
    """
    return ExpQ(-const(alpha) * (x ** 2 + y ** 2)
                - const(gamma) * (x - y) ** 2)


def kxs(omega, t1, t2):
    """
    Kernel between :math:`s` and :math:`x` processes.

    :param omega: :math:`\\omega`
    :param x: first variable
    :param y: second variable
    :return: exponentiated quadratic
    """
    return ExpQ(-const(omega) * (t1 - t2) ** 2)


class Factor:
    """
    Variable raised to some power.

    :param var: variable name
    :param power: power
    """

    def __init__(self, var, power):
        self._var = var
        self._power = power

    @property
    def var(self):
        return self._var

    @property
    def power(self):
        return self._power

    def eval(self, **var_map):
        """
        Evaluate factor.

        :param var_map: variable mapping
        :return: evaluated variable
        """
        return var_map[self._var] ** self._power

    def __eq__(self, other):
        return self._var == other.var and self._power == other.power

    def __str__(self):
        return '{}^{}'.format(self._var, self._power)

    def __mul__(self, other):
        if is_numeric(other) and other == 1:
            return self
        if not other.var == self._var:
            raise ValueError('other Factor must be function of same variable')
        return Factor(self.var, self.power + other.power)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __hash__(self):
        return hash(self._var) + hash(self._power)


class Term:
    """
    Product of a constant and multiple `Factor` objects.

    :param const: constant
    :param *factors: factors
    """

    def __init__(self, const, *factors):
        self._const = const
        # Discard factors if constant is zero
        if const == 0:
            factors = []
        # Merge common factors
        vars = set(x.var for x in factors)
        self._factors = set(reduce(operator.mul,
                                   filter(lambda y: y.var == var, factors),
                                   1) for var in vars)

    @property
    def factors(self):
        return self._factors

    @property
    def const(self):
        return self._const

    def is_function_of(self, var):
        """
        Check if this term is a function of some variable.

        :param var: variable name
        :return: boolean indicating whether this term is a function of `var`
        """
        return any([x.var == var for x in self._factors])

    def collect(self, factor):
        """
        Create a new term consisting of the same constant and all factors
        except one.

        :param factor: factor to exclude
        :return: term
        """
        if not factor in self._factors:
            raise RuntimeError('factor must be contained in term')
        return Term(self._const, *(self._factors - {factor}))

    def eval(self, **var_map):
        """
        Evaluate term.

        :param **var_map: variable mapping
        :return: evaluated term
        """
        return reduce(operator.mul,
                      [x.eval(**var_map) for x in self._factors],
                      self._const)

    def is_constant(self):
        """
        Check whether this term is constant.

        :return: boolean indicating whether this term is constant
        """
        return len(self.factors) == 0

    def substitute(self, var, poly):
        """
        Substitute a variable for a polynomial.

        :param var: variable name
        :param poly: polynomial
        :return: polynomial
        """
        factors = []
        power = 0
        for factor in self._factors:
            # Retain factor if its variable is different, otherwise save its
            # power to afterwards raise the polynomial to
            if factor.var == var:
                power = factor.power
            else:
                factors.append(factor)
        return Poly(Term(self._const, *factors)) * poly ** power

    def __str__(self):
        if len(self._factors) > 0:
            return '{} {}'.format(self._const, ' '.join(map(str,
                                                            self._factors)))
        else:
            return str(self._const)

    def __add__(self, other):
        if is_numeric(other) and other == 0:
            return self
        if not self._factors == other.factors:
            raise RuntimeError('factors must match')
        return Term(self._const + other.const, *self.factors)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if is_numeric(other) and other == 1:
            return self
        return Term(self._const * other.const,
                    *(list(self._factors) + list(other.factors)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        return self._const == other.const and self._factors == other.factors

    def __neg__(self):
        return Term(-self.const, *self.factors)

    def __hash__(self):
        return hash(self.const) + \
               reduce(operator.add, map(hash, self._factors), 0)


class Poly:
    """
    Sum of several `Term` objects.

    :param *terms: terms
    """

    def __init__(self, *terms):
        # Merge common terms
        factor_sets = set(frozenset(x.factors) for x in terms)
        self._terms = set(reduce(operator.add,
                                 filter(lambda y: y.factors == x, terms),
                                 0) for x in factor_sets)

    @property
    def terms(self):
        return self._terms

    def collect(self, factor):
        """
        Create a new polynomial consisting of terms whose factors contain
        factor and subsequently `Term.collect` `factor`.

        :param factor: factor
        :return: polynomial
        """
        return Poly(*[x.collect(factor)
                      for x in self._terms if factor in x.factors])

    def reject(self, var):
        """
        Create a new polynomial excluding terms whose factors contain the
        variable `var`.

        :param var: variable name
        :return: polynomial
        """
        return Poly(*[x for x in self._terms if not x.is_function_of(var)])

    def eval(self, **var_map):
        """
        Evaluate polynomial.

        :param **var_map: variable mapping
        :return: evaluated polynomial
        """
        return reduce(operator.add,
                      [x.eval(**var_map) for x in self._terms],
                      to_float(0))

    def is_constant(self):
        """
        Check whether the polynomial is constant.

        :return: boolean indicating whether the polynomial is constant
        """
        return len(self._terms) == 0 \
               or (len(self._terms) == 1
                   and list(self._terms)[0].is_constant())

    def substitute(self, var, poly):
        """
        Substitute a variable for a polynomial.

        :param var: variable name
        :param poly: polynomial
        :return: polynomial after substitution
        """
        return reduce(operator.add,
                      [x.substitute(var, poly) for x in self._terms],
                      0)

    def __str__(self):
        if len(self._terms) == 0:
            return '0'
        else:
            return ' + '.join([str(term) for term in self._terms])

    def __add__(self, other):
        if is_numeric(other) and other == 0:
            return self
        return Poly(*(list(self._terms) + list(other.terms)))

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return Poly(*[-term for term in self._terms])

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if is_numeric(other) and other == 1:
            return self
        return Poly(*[x * y for x in self._terms for y in other.terms])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power, modulo=None):
        if type(power) != int or power < 0:
            raise RuntimeError('can only raise to nonnegative integers')
        return reduce(operator.mul, [self for i in range(power)], 1)


class ExpQ:
    """
    A constant multiplied by an exponentiated polynomial.

    :param poly: polynomial
    :param const: constant
    """

    def __init__(self, poly, const=1):
        self._const = const
        self._poly = poly

    @property
    def poly(self):
        return self._poly

    @property
    def const(self):
        return self._const

    def substitute(self, var, poly):
        """
        Substitute a variable for a polynomial.

        :param var: variable name
        :param poly: polynomial
        :return: exponentiated quadratic form after substitution
        """
        return ExpQ(self._poly.substitute(var, poly), self._const)

    def eval(self, **var_map):
        """
        Evaluate exponentiated quadratic form.

        :param **var_map: variable mapping
        :return: evaluated exponentiated quadratic form
        """
        return tf.squeeze(self.const * tf.exp(self.poly.eval(**var_map)))

    def integrate(self, *vars, **var_map):
        """
        Integrate over a subset of the variables from :math:`-\\infty` to
        :math:`\\infty` and evaluate the result.

        :param *vars: variable names
        :param **var_map: variable mapping
        :return: result
        """
        eq = self
        for var in vars:
            eq = eq._integrate(var)
        return eq.eval(**var_map)

    def integrate_half(self, *vars, **var_map):
        """
        Integrate over a subset of the variables from :math:`-\\infty` to
        :math:`0` and evaluate the result.

        :param *vars: variable names
        :param **var_map: variable mapping
        :return: result
        """
        if len(vars) == 1:
            return self._integrate_half1(vars[0], **var_map)
        elif len(vars) == 2:
            return self._integrate_half2(*vars, **var_map)
        else:
            raise NotImplementedError()

    def integrate_box(self, *vars_and_lims, **var_map):
        """
        Integrate over a subset of the variables from some lower limit to some
        upper limit and evaluate the result. Infinity can be specified using
        `np.inf`. Any infinite lower limit corresponds to negative infinity,
        and any infinite upper limit corresponds to positive infinity.

        :param *vars_and_lims: triples containing the variable names, lower
                               limits, and upper limits
        :param **var_map: variable mapping
        :return: result
        """
        # Filter doubly infinite limits
        filtered_vars_and_lims = []
        expq = self
        for var, lower, upper in vars_and_lims:
            if is_inf(lower) and is_inf(upper):
                expq = expq._integrate(var)
            else:
                filtered_vars_and_lims.append((var, lower, upper))
        vars_and_lims = filtered_vars_and_lims

        # Return if all integration is done
        if len(vars_and_lims) == 0:
            return expq.eval(**var_map)

        # Integrate over box
        parts = [expq]
        for var, lower, upper in vars_and_lims:
            parts_new = []
            for part in parts:
                if not is_inf(upper):
                    parts_new.append(part.translate_var(var, upper))
                if not is_inf(lower):
                    parts_new.append(-part.translate_var(var, lower))
            parts = parts_new
        return reduce(operator.add,
                      [part.integrate_half(*zip(*vars_and_lims)[0], **var_map)
                       for part in parts],
                      to_float(0))

    def translate_var(self, var_name, poly):
        """
        Translate a variable by some polynomial; that is, substitute a variable
        for itself plus some polynomial.

        :param var_name: variable name
        :param poly: polynomial to shift by
        :return: exponentiated quadratic form
        """
        return ExpQ(self._poly.substitute(var_name, var(var_name) + poly),
                    self._const)

    def __mul__(self, other):
        return ExpQ(self.poly + other.poly,
                    self.const * other.const)

    def __neg__(self):
        return ExpQ(self.poly, -self.const)

    def _integrate(self, var):
        a = self._poly.collect(Factor(var, 2))
        b = self._poly.collect(Factor(var, 1))
        c = self._poly.reject(var)
        if not a.is_constant():
            raise ValueError('quadratic coefficient must be constant')
        a = a.eval()
        return ExpQ(Poly(Term(-.25 / a)) * b ** 2 + c,
                    self._const * (-np.pi / a) ** .5)

    def _integrate_half1(self, var, **var_map):
        a = self._poly.collect(Factor(var, 2))
        b = self._poly.collect(Factor(var, 1))
        c = self._poly.reject(var)
        if not a.is_constant():
            raise ValueError('quadratic coefficient must be constant')
        a, b, c = [x.eval(**var_map) for x in [a, b, c]]
        return tf.squeeze(.5 * self._const * (-np.pi / a) ** .5
                          * tf.exp(-.25 * b ** 2 / a + c)
                          * (1 - tf.erf(.5 * b / (-a) ** .5)))

    def _integrate_half2(self, var1, var2, **var_map):
        a11 = self._poly.collect(Factor(var1, 2))
        a22 = self._poly.collect(Factor(var2, 2))
        a12 = self._poly.collect(Factor(var1, 1)).collect(Factor(var2, 1))
        b1 = self._poly.collect(Factor(var1, 1)).reject(var2)
        b2 = self._poly.collect(Factor(var2, 1)).reject(var1)
        c = self._poly.reject(var1).reject(var2)

        # Evaluate
        if not (a11.is_constant() and a22.is_constant() and a12.is_constant()):
            raise ValueError('quadratic coefficients must be constant')
        a11, a22, a12 = [coef * x.eval()
                         for coef, x in zip([-2, -2, -1],
                                            [a11, a22, a12])]
        b1, b2 = [x.eval(**var_map) for x in [b1, b2]]
        c = c.eval(**var_map)

        # Determinant of A
        a_det = a11 * a22 - a12 ** 2

        # Inverse of A, corresponds to variance of distribution after
        # completing the square
        ia11 = a22 / a_det
        ia12 = -a12 / a_det
        ia22 = a11 / a_det

        # Mean of distribution after completing the square
        mu1 = ia11 * b1 + ia12 * b2
        mu2 = ia12 * b1 + ia22 * b2

        # Normalise and compute CDF part
        x1 = -mu1 / ia11 ** .5
        x2 = -mu2 / ia22 ** .5
        rho = ia12 / (ia11 * ia22) ** .5

        # Evaluate CDF for all `x1` and `x2`
        orig_shape = shape(mu1)
        num = reduce(operator.mul, orig_shape, 1)
        xs = tf.reshape(tf.stack([x1, x2], axis=-1), [num, 2])
        cdf_part = tf.reshape(bvn_cdf2(xs, rho), orig_shape)

        # Compute exponentiated part
        quad_form = .5 * (ia11 * b1 ** 2 + ia22 * b2 ** 2 + 2 * ia12 * b1 * b2)
        det_part = 2 * np.pi / a_det ** .5
        exp_part = tf.exp(quad_form + c) * det_part

        return tf.squeeze(self._const * cdf_part * exp_part)

    def __str__(self):
        if len(self._poly.terms) == 0:
            return str(self._const)
        else:
            return '{} exp({})'.format(self._const, str(self._poly))


if __name__ == '__main__':
    # Small test case. Probably should use units tests.

    t1, t2, t3 = var('t1'), var('t2'), var('t3')

    eq = ExpQ(- const(1) * t1 ** 2
              - const(2) * t2 ** 2
              - const(.5) * t1 * t2
              - const(2) * t1 * t3
              + const(3) * t2
              + const(4))
    var_map = {'t3': tf.constant(np.eye(2))}
    print sess.run(eq.integrate_half('t1', 't2', **var_map))
    print 'Mathematica\'s NIntegrate: 55.8181 and 11.7677'

    print sess.run(eq.integrate_box(('t1', const(-1), const(2)),
                                    ('t2', t3, const(3)),
                                    **var_map))
    print 'Mathematica\'s NIntegrate: 318.354 and 217.392'

    eq = ExpQ(const(-1) * t1 ** 2
              + const(-.5) * t1
              + const(4.))
    print sess.run(eq.integrate_half('t1'))
    print 'Mathematica\'s NIntegrate: 65.7397'
