# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2025
# File name: predicates/syntax.py

"""Syntactic handling of predicate-logic expressions."""

from __future__ import annotations
from functools import lru_cache
from typing import AbstractSet, Mapping, Optional, Sequence, Set, Tuple, Union

from logic_utils import fresh_variable_name_generator, frozen, \
                        memoized_parameterless_method

from propositions.syntax import Formula as PropositionalFormula, \
                                is_variable as is_propositional_variable

class ForbiddenVariableError(Exception):
    """Raised by `Term.substitute` and `Formula.substitute` when a substituted
    term contains a variable name that is forbidden in that context.

    Attributes:
        variable_name (`str`): the variable name that was forbidden in the
            context in which a term containing it was to be substituted.
    """
    variable_name: str

    def __init__(self, variable_name: str):
        """Initializes a `ForbiddenVariableError` from the offending variable
        name.

        Parameters:
            variable_name: variable name that is forbidden in the context in
                which a term containing it is to be substituted.
        """
        assert is_variable(variable_name)
        self.variable_name = variable_name

@lru_cache(maxsize=100)
def is_constant(string: str) -> bool:
    """Checks if the given string is a constant name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a constant name, ``False`` otherwise.
    """
    return (((string[0] >= '0' and string[0] <= '9') or \
              (string[0] >= 'a' and string[0] <= 'e')) and \
             string.isalnum()) or string == '_'

@lru_cache(maxsize=100)
def is_variable(string: str) -> bool:
    """Checks if the given string is a variable name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a variable name, ``False`` otherwise.
    """
    return string[0] >= 'u' and string[0] <= 'z' and string.isalnum()

@lru_cache(maxsize=100)
def is_function(string: str) -> bool:
    """Checks if the given string is a function name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a function name, ``False`` otherwise.
    """
    return string[0] >= 'f' and string[0] <= 't' and string.isalnum()

@frozen
class Term:
    """An immutable predicate-logic term in tree representation, composed from
    variable names and constant names, and function names applied to them.

    Attributes:
        root (`str`): the constant name, variable name, or function name at the
            root of the term tree.
        arguments (`~typing.Optional`\\[`~typing.Tuple`\\[`Term`, ...]]): the
            arguments of the root, if the root is a function name.
    """
    root: str
    arguments: Optional[Tuple[Term, ...]]

    def __init__(self, root: str, arguments: Optional[Sequence[Term]] = None):
        """Initializes a `Term` from its root and root arguments.

        Parameters:
            root: the root for the formula tree.
            arguments: the arguments for the root, if the root is a function
                name.
        """
        if is_constant(root) or is_variable(root):
            assert arguments is None
            self.root = root
        else:
            assert is_function(root)
            assert arguments is not None and len(arguments) > 0
            self.root = root
            self.arguments = tuple(arguments)

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current term.

        Returns:
            The standard string representation of the current term.
        """
        if is_constant(self.root) or is_variable(self.root):
            return self.root
        else:
            return self.root + '(' + ','.join(repr(arg) for arg in self.arguments) + ')'

    def __eq__(self, other: object) -> bool:
        """Compares the current term with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Term` object that equals the
            current term, ``False`` otherwise.
        """
        return isinstance(other, Term) and str(self) == str(other)
        
    def __ne__(self, other: object) -> bool:
        """Compares the current term with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Term` object or does not
            equal the current term, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Term, str]:
        """Parses a prefix of the given string into a term.

        Parameters:
            string: string to parse, which has a prefix that is a valid
                representation of a term.

        Returns:
            A pair of the parsed term and the unparsed suffix of the string. If
            the given string has as a prefix a constant name (e.g., ``'c12'``)
            or a variable name (e.g., ``'x12'``), then the parsed prefix will be
            that entire name (and not just a part of it, such as ``'x1'``).
        """
        i = 0
        while i < len(string) and string[i].isalnum():
            i += 1
        name = string[:i]
        if i < len(string) and string[i] == '(':
            i += 1
            args = []
            rest = string[i:]
            while True:
                arg, rest = Term._parse_prefix(rest)
                args.append(arg)
                if not rest or rest[0] != ',':
                    break
                rest = rest[1:]
            if not rest or rest[0] != ')':
                raise ValueError("Missing closing parenthesis")
            rest = rest[1:]
            return Term(name, args), rest
        else:
            return Term(name), string[i:]

    @staticmethod
    def parse(string: str) -> Term:
        """Parses the given valid string representation into a term.

        Parameters:
            string: string to parse.

        Returns:
            A term whose standard string representation is the given string.
        """
        term, rest = Term._parse_prefix(string)
        if rest != '':
            raise ValueError("Extra characters")
        return term

    def constants(self) -> Set[str]:
        """Finds all constant names in the current term.

        Returns:
            A set of all constant names used in the current term.
        """
        if is_constant(self.root):
            return {self.root}
        if is_variable(self.root):
            return set()
        result = set()
        for arg in self.arguments:
            result.update(arg.constants())
        return result

    def variables(self) -> Set[str]:
        """Finds all variable names in the current term.

        Returns:
            A set of all variable names used in the current term.
        """
        if is_variable(self.root):
            return {self.root}
        if is_constant(self.root):
            return set()
        result = set()
        for arg in self.arguments:
            result.update(arg.variables())
        return result

    def functions(self) -> Set[Tuple[str, int]]:
        """Finds all function names in the current term, along with their
        arities.

        Returns:
            A set of pairs of function name and arity (number of arguments) for
            all function names used in the current term.
        """
        result = set()
        if is_function(self.root):
            result.add((self.root, len(self.arguments)))
        for arg in self.arguments:
            result.update(arg.functions())
        return result

    def substitute(self, substitution_map: Mapping[str, Term],
                   forbidden_variables: AbstractSet[str] = frozenset()) -> Term:
        """Substitutes in the current term, each constant name `construct` or
        variable name `construct` that is a key in `substitution_map` with the
        term `substitution_map`\\ ``[``\\ `construct`\\ ``]``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.
            forbidden_variables: variable names not allowed in substitution
                terms.

        Returns:
            The term resulting from performing all substitutions. Only
            constant name and variable name occurrences originating in the
            current term are substituted (i.e., those originating in one of the
            specified substitutions are not subjected to additional
            substitutions).

        Raises:
            ForbiddenVariableError: If a term that is used in the requested
                substitution contains a variable name from
                `forbidden_variables`.

        Examples:
            >>> Term.parse('f(x,c)').substitute(
            ...     {'c': Term.parse('plus(d,x)'), 'x': Term.parse('c')}, {'y'})
            f(c,plus(d,x))

            >>> Term.parse('f(x,c)').substitute(
            ...     {'c': Term.parse('plus(d,y)')}, {'y'})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: y
        """
        for construct in substitution_map:
            assert is_constant(construct) or is_variable(construct)
        for variable in forbidden_variables:
            assert is_variable(variable)
        raise NotImplementedError

@lru_cache(maxsize=100)
def is_equality(string: str) -> bool:
    """Checks if the given string is the equality relation.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is the equality relation, ``False``
        otherwise.
    """
    return string == '='

@lru_cache(maxsize=100)
def is_relation(string: str) -> bool:
    """Checks if the given string is a relation name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a relation name, ``False`` otherwise.
    """
    return string[0] >= 'F' and string[0] <= 'T' and string.isalnum()

@lru_cache(maxsize=100)
def is_unary(string: str) -> bool:
    """Checks if the given string is a unary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a unary operator, ``False`` otherwise.
    """
    return string == '~'

@lru_cache(maxsize=100)
def is_binary(string: str) -> bool:
    """Checks if the given string is a binary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a binary operator, ``False`` otherwise.
    """
    return string == '&' or string == '|' or string == '->'

@lru_cache(maxsize=100)
def is_quantifier(string: str) -> bool:
    """Checks if the given string is a quantifier.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a quantifier, ``False`` otherwise.
    """
    return string == 'A' or string == 'E'

@frozen
class Formula:
    """An immutable predicate-logic formula in tree representation, composed
    from relation names applied to predicate-logic terms, and operators and
    quantifications applied to them.

    Attributes:
        root (`str`): the relation name, equality relation, operator, or
            quantifier at the root of the formula tree.
        arguments (`~typing.Optional`\\[`~typing.Tuple`\\[`Term`, ...]]): the
            arguments of the root, if the root is a relation name or the
            equality relation.
        first (`~typing.Optional`\\[`Formula`]): the first operand of the root,
            if the root is a unary or binary operator.
        second (`~typing.Optional`\\[`Formula`]): the second operand of the
            root, if the root is a binary operator.
        variable (`~typing.Optional`\\[`str`]): the variable name quantified by
            the root, if the root is a quantification.
        statement (`~typing.Optional`\\[`Formula`]): the statement quantified by
            the root, if the root is a quantification.
    """
    root: str
    arguments: Optional[Tuple[Term, ...]]
    first: Optional[Formula]
    second: Optional[Formula]
    variable: Optional[str]
    statement: Optional[Formula]

    def __init__(self, root: str,
                 arguments_or_first_or_variable: Union[Sequence[Term],
                                                       Formula, str],
                 second_or_statement: Optional[Formula] = None):
        """Initializes a `Formula` from its root and root arguments, root
        operands, or root quantified variable name and statement.

        Parameters:
            root: the root for the formula tree.
            arguments_or_first_or_variable: the arguments for the root, if the
                root is a relation name or the equality relation; the first
                operand for the root, if the root is a unary or binary operator;
                the variable name to be quantified by the root, if the root is a
                quantification.
            second_or_statement: the second operand for the root, if the root is
                a binary operator; the statement to be quantified by the root,
                if the root is a quantification.
        """
        if is_equality(root) or is_relation(root):
            assert isinstance(arguments_or_first_or_variable, Sequence) and \
                   not isinstance(arguments_or_first_or_variable, str)
            if is_equality(root):
                assert len(arguments_or_first_or_variable) == 2
            assert second_or_statement is None
            self.root, self.arguments = \
                root, tuple(arguments_or_first_or_variable)
        elif is_unary(root):
            assert isinstance(arguments_or_first_or_variable, Formula)
            assert second_or_statement is None
            self.root, self.first = root, arguments_or_first_or_variable
        elif is_binary(root):
            assert isinstance(arguments_or_first_or_variable, Formula)
            assert second_or_statement is not None
            self.root, self.first, self.second = \
                root, arguments_or_first_or_variable, second_or_statement
        else:
            assert is_quantifier(root)
            assert isinstance(arguments_or_first_or_variable, str) and \
                   is_variable(arguments_or_first_or_variable)
            assert second_or_statement is not None
            self.root, self.variable, self.statement = \
                root, arguments_or_first_or_variable, second_or_statement

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current formula.

        Returns:
            The standard string representation of the current formula.
        """
        if is_equality(self.root):
            return f"{self.arguments[0]}={self.arguments[1]}"
        if is_relation(self.root):
            return f"{self.root}({','.join(str(arg) for arg in self.arguments)})"
        if is_unary(self.root):
            sub = self.first
            sub_str = str(sub)
            if is_binary(sub.root) or is_unary(sub.root) or is_quantifier(sub.root):
                sub_str = f"({sub_str})"
            return f"~{sub_str}"
        if is_binary(self.root):
            left, right = self.first, self.second
            left_str = str(left)
            if is_binary(left.root) or is_unary(left.root) or is_quantifier(left.root):
                if self._needs_parens(left, self.root):
                    left_str = f"({left_str})"
            right_str = str(right)
            if is_binary(right.root) or is_unary(right.root) or is_quantifier(right.root):
                if self._needs_parens(right, self.root):
                    right_str = f"({right_str})"
            result = f"{left_str}{self.root}{right_str}"
            if self.root == '->':
                result = f"({result})"
            return result
        if is_quantifier(self.root):
            return f"{self.root}{self.variable}[{self.statement}]"
        raise ValueError("Invalid formula")

    def _needs_parens(self, sub: 'Formula', parent_root: str) -> bool:
        prec = {'->':1, '|':2, '&':3, '~':4, 'A':4, 'E':4}
        sub_prec = prec.get(sub.root, 0)
        parent_prec = prec.get(parent_root, 0)
        if sub_prec < parent_prec:
            return True
        if sub_prec == parent_prec and parent_root == '->' and sub.root == '->':
            return True
        return False

    def __eq__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Formula` object that equals the
            current formula, ``False`` otherwise.
        """
        return isinstance(other, Formula) and str(self) == str(other)
        
    def __ne__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Formula` object or does not
            equal the current formula, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Formula, str]:
        """Parses a prefix of the given string into a formula.

        Parameters:
            string: string to parse, which has a prefix that is a valid
                representation of a formula.

        Returns:
            A pair of the parsed formula and the unparsed suffix of the string.
            If the given string has as a prefix a term followed by an equality
            followed by a constant name (e.g., ``'f(y)=c12'``) or by a variable
            name (e.g., ``'f(y)=x12'``), then the parsed prefix will include
            that entire name (and not just a part of it, such as ``'f(y)=x1'``).
        """
        def parse_identifier(s):
            i = 0
            while i < len(s) and s[i].isalnum():
                i += 1
            return s[:i], s[i:]

        def parse_term(s):
            return Term._parse_prefix(s)

        def parse_arguments(s):
            terms = []
            term, rest = parse_term(s)
            terms.append(term)
            while rest and rest[0] == ',':
                rest = rest[1:]
                term, rest = parse_term(rest)
                terms.append(term)
            return terms, rest

        def parse_atomic(s):
            if s and s[0] == '(':
                formula, rest = parse_implication(s[1:])
                if not rest or rest[0] != ')':
                    raise ValueError("Missing closing parenthesis")
                return formula, rest[1:]
            if s and s[0] >= 'F' and s[0] <= 'T':
                name, rest = parse_identifier(s)
                if rest and rest[0] == '(':
                    rest = rest[1:]
                    args, rest = parse_arguments(rest)
                    if not rest or rest[0] != ')':
                        raise ValueError("Missing closing parenthesis")
                    rest = rest[1:]
                    return Formula(name, args), rest
                else:
                    raise ValueError("Expected relation with arguments")
            term1, rest = parse_term(s)
            if rest and rest[0] == '=':
                rest = rest[1:]
                term2, rest = parse_term(rest)
                return Formula('=', (term1, term2)), rest
            raise ValueError("Expected atomic formula")

        def parse_quantifier_or_atomic(s):
            if s and (s[0] == 'A' or s[0] == 'E'):
                quant = s[0]
                rest = s[1:]
                var, rest = parse_identifier(rest)
                if not is_variable(var):
                    raise ValueError("Expected variable after quantifier")
                if not rest or rest[0] != '[':
                    raise ValueError("Expected '[' after quantifier")
                rest = rest[1:]
                statement, rest = parse_implication(rest)
                if not rest or rest[0] != ']':
                    raise ValueError("Expected ']' after statement")
                rest = rest[1:]
                return Formula(quant, var, statement), rest
            else:
                return parse_atomic(s)

        def parse_unary(s):
            if s and s[0] == '~':
                sub, rest = parse_unary(s[1:])
                return Formula('~', sub), rest
            else:
                return parse_quantifier_or_atomic(s)

        def parse_conjunction(s):
            left, rest = parse_unary(s)
            while rest and rest[0] == '&':
                rest = rest[1:]
                right, rest = parse_unary(rest)
                left = Formula('&', left, right)
            return left, rest

        def parse_disjunction(s):
            left, rest = parse_conjunction(s)
            while rest and rest[0] == '|':
                rest = rest[1:]
                right, rest = parse_conjunction(rest)
                left = Formula('|', left, right)
            return left, rest

        def parse_implication(s):
            left, rest = parse_disjunction(s)
            while rest and rest.startswith('->'):
                rest = rest[2:]
                right, rest = parse_disjunction(rest)
                left = Formula('->', left, right)
            return left, rest

        formula, rest = parse_implication(string)
        return formula, rest

    @staticmethod
    def parse(string: str) -> Formula:
        """Parses the given valid string representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose standard string representation is the given string.
        """
        formula, rest = Formula._parse_prefix(string)
        if rest != '':
            raise ValueError("Extra characters")
        return formula

    def constants(self) -> Set[str]:
        """Finds all constant names in the current formula.

        Returns:
            A set of all constant names used in the current formula.
        """
        if is_equality(self.root) or is_relation(self.root):
            result = set()
            for arg in self.arguments:
                result.update(arg.constants())
            return result
        if is_unary(self.root):
            return self.first.constants()
        if is_binary(self.root):
            return self.first.constants().union(self.second.constants())
        if is_quantifier(self.root):
            return self.statement.constants()
        raise ValueError("Invalid formula")

    def variables(self) -> Set[str]:
        """Finds all variable names in the current formula.

        Returns:
            A set of all variable names used in the current formula.
        """
        if is_equality(self.root) or is_relation(self.root):
            result = set()
            for arg in self.arguments:
                result.update(arg.variables())
            return result
        if is_unary(self.root):
            return self.first.variables()
        if is_binary(self.root):
            return self.first.variables().union(self.second.variables())
        if is_quantifier(self.root):
            result = self.statement.variables()
            result.add(self.variable)
            return result
        raise ValueError("Invalid formula")

    def free_variables(self) -> Set[str]:
        """Finds all variable names that are free in the current formula.

        Returns:
            A set of every variable name that is used in the current formula not
            only within a scope of a quantification on that variable name.
        """
        if is_equality(self.root) or is_relation(self.root):
            result = set()
            for arg in self.arguments:
                result.update(arg.variables())
            return result
        if is_unary(self.root):
            return self.first.free_variables()
        if is_binary(self.root):
            return self.first.free_variables().union(self.second.free_variables())
        if is_quantifier(self.root):
            fv = self.statement.free_variables()
            fv.discard(self.variable)
            return fv
        raise ValueError("Invalid formula")

    def functions(self) -> Set[Tuple[str, int]]:
        """Finds all function names in the current formula, along with their
        arities.

        Returns:
            A set of pairs of function name and arity (number of arguments) for
            all function names used in the current formula.
        """
        if is_equality(self.root) or is_relation(self.root):
            result = set()
            for arg in self.arguments:
                result.update(arg.functions())
            return result
        if is_unary(self.root):
            return self.first.functions()
        if is_binary(self.root):
            return self.first.functions().union(self.second.functions())
        if is_quantifier(self.root):
            return self.statement.functions()
        raise ValueError("Invalid formula")

    def relations(self) -> Set[Tuple[str, int]]:
        """Finds all relation names in the current formula, along with their
        arities.

        Returns:
            A set of pairs of relation name and arity (number of arguments) for
            all relation names used in the current formula.
        """
        if is_relation(self.root):
            result = {(self.root, len(self.arguments))}
        else:
            result = set()
        if is_equality(self.root):
            pass
        elif is_unary(self.root):
            result.update(self.first.relations())
        elif is_binary(self.root):
            result.update(self.first.relations())
            result.update(self.second.relations())
        elif is_quantifier(self.root):
            result.update(self.statement.relations())
        return result

    def substitute(self, substitution_map: Mapping[str, Term],
                   forbidden_variables: AbstractSet[str] = frozenset()) -> \
            Formula:
        """Substitutes in the current formula, each constant name `construct` or
        free occurrence of variable name `construct` that is a key in
        `substitution_map` with the term
        `substitution_map`\\ ``[``\\ `construct`\\ ``]``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.
            forbidden_variables: variable names not allowed in substitution
                terms.

        Returns:
            The formula resulting from performing all substitutions. Only
            constant name and variable name occurrences originating in the
            current formula are substituted (i.e., those originating in one of
            the specified substitutions are not subjected to additional
            substitutions).

        Raises:
            ForbiddenVariableError: If a term that is used in the requested
                substitution contains a variable name from `forbidden_variables`
                or a variable name occurrence that becomes bound when that term
                is substituted into the current formula.

        Examples:
            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,x)'), 'x': Term.parse('c')}, {'z'})
            Ay[c=plus(d,x)]

            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,z)')}, {'z'})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: z

            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,y)')})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: y
        """
        for construct in substitution_map:
            assert is_constant(construct) or is_variable(construct)
        for variable in forbidden_variables:
            assert is_variable(variable)
        raise NotImplementedError

    def propositional_skeleton(self) -> Tuple[PropositionalFormula,
                                              Mapping[str, Formula]]:
        """Computes a propositional skeleton of the current formula.

        Returns:
            A pair. The first element of the pair is a propositional formula
            obtained from the current formula by substituting every (outermost)
            subformula that has a relation name, equality, or quantifier at its
            root with a propositional variable name, consistently such that
            multiple identical such (outermost) subformulas are substituted with
            the same propositional variable name. The propositional variable
            names used for substitution are obtained, from left to right
            (considering their first occurrence), by calling
            `next`\\ ``(``\\ `~logic_utils.fresh_variable_name_generator`\\ ``)``.
            The second element of the pair is a mapping from each propositional
            variable name to the subformula for which it was substituted.

        Examples:
            >>> formula = Formula.parse('((Ax[x=7]&x=7)|(~Q(y)->x=7))')
            >>> formula.propositional_skeleton()
            (((z1&z2)|(~z3->z2)), {'z1': Ax[x=7], 'z2': x=7, 'z3': Q(y)})
            >>> formula.propositional_skeleton()
            (((z4&z5)|(~z6->z5)), {'z4': Ax[x=7], 'z5': x=7, 'z6': Q(y)})
        """
        raise NotImplementedError

    @staticmethod
    def from_propositional_skeleton(skeleton: PropositionalFormula,
                                    substitution_map: Mapping[str, Formula]) \
            -> Formula:
        """Computes a predicate-logic formula from a propositional skeleton and
        a substitution map.

        Arguments:
            skeleton: propositional skeleton for the formula to compute,
                containing no constants or operators beyond ``'~'``, ``'->'``,
                ``'|'``, and ``'&'``.
            substitution_map: mapping from each propositional variable name of
                the given propositional skeleton to a predicate-logic formula.

        Returns:
            A predicate-logic formula obtained from the given propositional
            skeleton by substituting each propositional variable name with the
            formula mapped to it by the given map.

        Examples:
            >>> Formula.from_propositional_skeleton(
            ...     PropositionalFormula.parse('((z1&z2)|(~z3->z2))'),
            ...     {'z1': Formula.parse('Ax[x=7]'), 'z2': Formula.parse('x=7'),
            ...      'z3': Formula.parse('Q(y)')})
            ((Ax[x=7]&x=7)|(~Q(y)->x=7))

            >>> Formula.from_propositional_skeleton(
            ...     PropositionalFormula.parse('((z9&z2)|(~z3->z2))'),
            ...     {'z2': Formula.parse('x=7'), 'z3': Formula.parse('Q(y)'),
            ...      'z9': Formula.parse('Ax[x=7]')})
            ((Ax[x=7]&x=7)|(~Q(y)->x=7))
        """
        for operator in skeleton.operators():
            assert is_unary(operator) or is_binary(operator)
        for variable in skeleton.variables():
            assert variable in substitution_map
        raise NotImplementedError