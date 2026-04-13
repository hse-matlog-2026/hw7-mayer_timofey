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
    variable_name: str

    def __init__(self, variable_name: str):
        assert is_variable(variable_name)
        self.variable_name = variable_name

@lru_cache(maxsize=100)
def is_constant(string: str) -> bool:
    return (((string[0] >= '0' and string[0] <= '9') or \
              (string[0] >= 'a' and string[0] <= 'e')) and \
             string.isalnum()) or string == '_'

@lru_cache(maxsize=100)
def is_variable(string: str) -> bool:
    return string[0] >= 'u' and string[0] <= 'z' and string.isalnum()

@lru_cache(maxsize=100)
def is_function(string: str) -> bool:
    return string[0] >= 'f' and string[0] <= 't' and string.isalnum()

@frozen
class Term:
    root: str
    arguments: Optional[Tuple[Term, ...]]

    def __init__(self, root: str, arguments: Optional[Sequence[Term]] = None):
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
        return isinstance(other, Term) and str(self) == str(other)
        
    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Term, str]:
        if not string:
            raise ValueError("Empty string")
        if string[0] == '_':
            return Term('_'), string[1:]
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
        raise NotImplementedError

@lru_cache(maxsize=100)
def is_equality(string: str) -> bool:
    return string == '='

@lru_cache(maxsize=100)
def is_relation(string: str) -> bool:
    return string[0] >= 'F' and string[0] <= 'T' and string.isalnum()

@lru_cache(maxsize=100)
def is_unary(string: str) -> bool:
    return string == '~'

@lru_cache(maxsize=100)
def is_binary(string: str) -> bool:
    return string == '&' or string == '|' or string == '->'

@lru_cache(maxsize=100)
def is_quantifier(string: str) -> bool:
    return string == 'A' or string == 'E'

@frozen
class Formula:
    root: str
    arguments: Optional[Tuple[Term, ...]]
    first: Optional[Formula]
    second: Optional[Formula]
    variable: Optional[str]
    statement: Optional[Formula]
    _parenthesized: bool

    def __init__(self, root: str,
                 arguments_or_first_or_variable: Union[Sequence[Term],
                                                       Formula, str],
                 second_or_statement: Optional[Formula] = None,
                 _parenthesized: bool = False):
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
        self._parenthesized = _parenthesized

    def _with_parentheses(self):
        if is_equality(self.root) or is_relation(self.root):
            return Formula(self.root, self.arguments, _parenthesized=True)
        elif is_unary(self.root):
            return Formula(self.root, self.first, _parenthesized=True)
        elif is_binary(self.root):
            return Formula(self.root, self.first, self.second, _parenthesized=True)
        elif is_quantifier(self.root):
            return Formula(self.root, self.variable, self.statement, _parenthesized=True)
        else:
            raise ValueError("Invalid formula")

    @memoized_parameterless_method
    def __repr__(self) -> str:
        if is_equality(self.root):
            s = f"{self.arguments[0]}={self.arguments[1]}"
        elif is_relation(self.root):
            s = f"{self.root}({','.join(str(arg) for arg in self.arguments)})"
        elif is_unary(self.root):
            sub = self.first
            sub_str = str(sub)
            if is_binary(sub.root) or is_unary(sub.root) or is_quantifier(sub.root):
                sub_str = f"({sub_str})"
            s = f"~{sub_str}"
        elif is_binary(self.root):
            left, right = self.first, self.second
            left_str = str(left)
            if is_binary(left.root) or is_unary(left.root) or is_quantifier(left.root):
                if self._needs_parens(left, self.root):
                    left_str = f"({left_str})"
            right_str = str(right)
            if is_binary(right.root) or is_unary(right.root) or is_quantifier(right.root):
                if self._needs_parens(right, self.root):
                    right_str = f"({right_str})"
            s = f"{left_str}{self.root}{right_str}"
        elif is_quantifier(self.root):
            s = f"{self.root}{self.variable}[{self.statement}]"
        else:
            raise ValueError("Invalid formula")
        if self._parenthesized:
            s = f"({s})"
        return s

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
        return isinstance(other, Formula) and str(self) == str(other)
        
    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Formula, str]:
        def parse_identifier(s):
            i = 0
            while i < len(s) and (s[i].isalnum() or s[i] == '_'):
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
                rest = rest[1:]
                formula = formula._with_parentheses()
                return formula, rest
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
                   forbidden_variables: AbstractSet[str] = frozenset()) -> Formula:
        for construct in substitution_map:
            assert is_constant(construct) or is_variable(construct)
        for variable in forbidden_variables:
            assert is_variable(variable)
        raise NotImplementedError

    def propositional_skeleton(self) -> Tuple[PropositionalFormula,
                                              Mapping[str, Formula]]:
        raise NotImplementedError

    @staticmethod
    def from_propositional_skeleton(skeleton: PropositionalFormula,
                                    substitution_map: Mapping[str, Formula]) -> Formula:
        for operator in skeleton.operators():
            assert is_unary(operator) or is_binary(operator)
        for variable in skeleton.variables():
            assert variable in substitution_map
        raise NotImplementedError