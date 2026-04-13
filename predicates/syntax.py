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

@lru_cache(maxsize=100) # Cache the return value of is_constant
def is_constant(string: str) -> bool:
    """Checks if the given string is a constant name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a constant name, ``False`` otherwise.
    """
    return  (((string[0] >= '0' and string[0] <= '9') or \
              (string[0] >= 'a' and string[0] <= 'e')) and \
             string.isalnum()) or string == '_'

@lru_cache(maxsize=100) # Cache the return value of is_variable
def is_variable(string: str) -> bool:
    """Checks if the given string is a variable name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a variable name, ``False`` otherwise.
    """
    return string[0] >= 'u' and string[0] <= 'z' and string.isalnum()

@lru_cache(maxsize=100) # Cache the return value of is_function
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
        term, rest = Term._parse_prefix(string)
        if rest != '':
            raise ValueError("Extra characters")
        return term

    def constants(self) -> Set[str]:
        if is_constant(self.root):
            return {self.root}
        if is_variable(self.root):
            return set()
        result = set()
        for arg in self.arguments:
            result.update(arg.constants())
        return result

    def variables(self) -> Set[str]:
        if is_variable(self.root):
            return {self.root}
        if is_constant(self.root):
            return set()
        result = set()
        for arg in self.arguments:
            result.update(arg.variables())
        return result

    def functions(self) -> Set[Tuple[str, int]]:
        result = set()
        if is_function(self.root):
            result.add((self.root, len(self.arguments)))
        for arg in self.arguments:
            result.update(arg.functions())
        return result

    def substitute(self, substitution_map: Mapping[str, Term],
                   forbidden_variables: AbstractSet[str] = frozenset()) -> Term:
        for construct in substitution_map:
            assert is_constant(construct) or is_variable(construct)
        for variable in forbidden_variables:
            assert is_variable(variable)
        # Task 9.1

@lru_cache(maxsize=100) # Cache the return value of is_equality
def is_equality(string: str) -> bool:
    """Checks if the given string is the equality relation.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is the equality relation, ``False``
        otherwise.
    """
    return string == '='

@lru_cache(maxsize=100) # Cache the return value of is_relation
def is_relation(string: str) -> bool:
    """Checks if the given string is a relation name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a relation name, ``False`` otherwise.
    """
    return string[0] >= 'F' and string[0] <= 'T' and string.isalnum()

@lru_cache(maxsize=100) # Cache the return value of is_unary
def is_unary(string: str) -> bool:
    """Checks if the given string is a unary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a unary operator, ``False`` otherwise.
    """
    return string == '~'

@lru_cache(maxsize=100) # Cache the return value of is_binary
def is_binary(string: str) -> bool:
    """Checks if the given string is a binary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a binary operator, ``False`` otherwise.
    """
    return string == '&' or string == '|' or string == '->'

@lru_cache(maxsize=100) # Cache the return value of is_quantifier
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
            # Populate self.root and self.arguments
            assert isinstance(arguments_or_first_or_variable, Sequence) and \
                   not isinstance(arguments_or_first_or_variable, str)
            if is_equality(root):
                assert len(arguments_or_first_or_variable) == 2
            assert second_or_statement is None
            self.root, self.arguments = \
                root, tuple(arguments_or_first_or_variable)
        elif is_unary(root):
            # Populate self.first
            assert isinstance(arguments_or_first_or_variable, Formula)
            assert second_or_statement is None
            self.root, self.first = root, arguments_or_first_or_variable
        elif is_binary(root):
            # Populate self.first and self.second
            assert isinstance(arguments_or_first_or_variable, Formula)
            assert second_or_statement is not None
            self.root, self.first, self.second = \
                root, arguments_or_first_or_variable, second_or_statement
        else:
            assert is_quantifier(root)
            # Populate self.variable and self.statement
            assert isinstance(arguments_or_first_or_variable, str) and \
                   is_variable(arguments_or_first_or_variable)
            assert second_or_statement is not None
            self.root, self.variable, self.statement = \
                root, arguments_or_first_or_variable, second_or_statement

    @memoized_parameterless_method
    def __repr__(self) -> str:
        def priority(r):
            if r in ('A','E','~'):
                return 3
            if r == '&':
                return 2
            if r == '|':
                return 1
            if r == '->':
                return 0
            return 4

        if is_equality(self.root):
            return f"{self.arguments[0]}={self.arguments[1]}"
        if is_relation(self.root):
            return f"{self.root}({','.join(str(arg) for arg in self.arguments)})"
        if is_unary(self.root):
            sub = self.first
            if priority(sub.root) < priority(self.root):
                return f"~({sub})"
            else:
                return f"~{sub}"
        if is_binary(self.root):
            left, right = self.first, self.second
            left_str = str(left) if priority(left.root) >= priority(self.root) else f"({left})"
            right_str = str(right) if priority(right.root) > priority(self.root) else f"({right})"
            return f"{left_str}{self.root}{right_str}"
        if is_quantifier(self.root):
            return f"{self.root}{self.variable}[{self.statement}]"
        raise ValueError("Invalid formula")

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
        formula, rest = Formula._parse_prefix(string)
        if rest != '':
            raise ValueError("Extra characters")
        return formula

    def constants(self) -> Set[str]:
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
        for construct in substitution_map:
            assert is_constant(construct) or is_variable(construct)
        for variable in forbidden_variables:
            assert is_variable(variable)
        # Task 9.2

    def propositional_skeleton(self) -> Tuple[PropositionalFormula,
                                              Mapping[str, Formula]]:
        # Task 9.8

    @staticmethod
    def from_propositional_skeleton(skeleton: PropositionalFormula,
                                    substitution_map: Mapping[str, Formula]) \
            -> Formula:
        # Task 9.10