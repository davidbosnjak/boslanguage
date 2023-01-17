#######################################
# IMPORTS
#######################################
import string
#######################################
# CONSTANTS
#######################################

DIGITS = "0123456789"

LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS+DIGITS
#######################################
# ERRORS
#######################################


class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f"{self.error_name}: {self.details}\n"
        result += f"File {self.pos_start.fn}, line {self.pos_start.ln + 1}"
        result += "\n\n" + string_with_arrows(
            self.pos_start.ftxt, self.pos_start, self.pos_end
        )
        return result


class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, "Illegal Character", details)


class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=""):
        super().__init__(pos_start, pos_end, "Invalid Syntax", details)
class RunTimeError(Error):
    def __init__(self, pos_start, pos_end, description, context):
        super().__init__(pos_start, pos_end, "Runtime Error: ",description)
        self.context = context
    
    def as_string(self):
        result = self.generate_traceback()
        result += f"{self.error_name}: {self.details}\n"
        result += "\n\n" + string_with_arrows(
            self.pos_start.ftxt, self.pos_start, self.pos_end
        )
        return result
    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        context = self.context
        while context:
            result = f'File {pos.fn}, line{str(pos.ln+1)}, in {context.display_name}\n'
            pos = context.parent_entry_pos
            context = context.parent
        return 'Traceback (most recent call last):\n'+result

            

#######################################
# POSITION
#######################################


class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == "\n":
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)


#######################################
# TOKENS
#######################################

TT_INT = "INT"
TT_FLOAT = "FLOAT"
TT_PLUS = "PLUS"
TT_MINUS = "MINUS"
TT_MUL = "MUL"
TT_DIV = "DIV"
TT_LPAREN = "LPAREN"
TT_RPAREN = "RPAREN"
TT_EOF = "EOF"
TT_INT = "INT"
TT_LITERAL = "LITERAL"
TT_IDENTIFIER = "IDENTIFIER"
TT_ASSIGNMENT = "EQUALS"
TT_TYPEKEYWORD = "TYPEKEYWORD"
TT_POWER = "POWER"
TT_DOUBLEQUOTE = "DOUBLEQUOTE"
TT_SINGLEQUOTE = "SINGLEQUOTE"
TT_SEMICOLON = "SEMICOLON"
TT_CHAR = "CHAR"
TT_COMPARE = "COMPARE"
TT_BOOL = "BOOL"
TT_KEYWORD = "KEYWORD"
TT_BUILTFUNC = "BUILTFUNC"
TT_GREATERTHAN = "GREATERTHAN"
TT_LESSTHAN = "LESSTHAN"
TT_GREATERTHANEQ = "GREATERTHANEQ"
TT_LESSTHANEQ = "LESSTHANEQ"
TT_NOTEQUALTO = "NOTEQUALTO"
TT_COMMENT = "COMMENT"
TT_AND = "and"
TT_OR = "or"
TT_XOR = "xor"
TT_NOT = "not"



#values for TYPEKEYWORD
typeValues = ["char","string", "int", "bool"]

#values for KEYWORD
keywords = ["if", "for", "while", "elif", "else"]

#values for BUILTFUNC
builtFuncs = ["print"]

class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end

    def __repr__(self):
        if self.value:
            return f"{self.type}:{self.value}"
        return f"{self.type}"
    def matches(self, type_, value):
        return self.type == type_ and self.value == value
    def matchesList(self, type_, listValue):
        if self.value in listValue and self.type == type_: return True
        else: return False


#######################################
# LEXER
#######################################


class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = (
            self.text[self.pos.idx] if self.pos.idx < len(self.text) else None
        )

    def make_tokens(self):
        tokens = []

        while self.current_char != None:
            if self.current_char in " \t":
                self.advance()
            elif self.current_char == "'":
                self.advance()
                tokens.append(Token(TT_CHAR, self.current_char, pos_start = self.pos))
                self.advance()
                self.advance()
            elif self.current_char == "!":
                self.advance()
                if self.current_char == "=":
                    tokens.append(Token(TT_NOTEQUALTO, pos_start=self.pos))
                    self.advance()
            elif self.current_char ==">":
                self.advance()
                if self.current_char!= "=":

                    tokens.append(Token(TT_GREATERTHAN, pos_start=self.pos))
                else:
                    tokens.append(Token(TT_GREATERTHANEQ, pos_start=self.pos))
                    self.advance()
            elif self.current_char == "<":
                self.advance()
                if self.current_char!="=":
                    tokens.append(Token(TT_LESSTHAN, pos_start=self.pos))
                else: 
                    tokens.append(Token(TT_GREATERTHANEQ, pos_start=self.pos))
                    self.advance()
            
            elif self.current_char == ";":
                tokens.append(Token(TT_SEMICOLON, pos_start=self.pos))
            elif self.current_char in LETTERS:
                tokens.append(self.make_word())
            elif self .current_char == '"':
                tokens.append(self.make_literal())
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char == "+":
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == "^":
                tokens.append(Token(TT_POWER, pos_start =self.pos))
                self.advance()
            elif self.current_char == "-":
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == "*":
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == "/":
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == "(":
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ")":
                start_pos = self.pos.copy()
                self.advance()

                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == "=":
                self.advance()
                if(self.current_char !="="):
                    tokens.append(Token(TT_ASSIGNMENT, pos_start=self.pos))
                else:
                    tokens.append(Token(TT_COMPARE, pos_start=self.pos))
                    self.advance()
                

            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None
    def make_word(self):
        word_str= ""
        pos_start = self.pos.copy()
        while self.current_char !=None and self.current_char in LETTERS +'_':
            word_str+=self.current_char
            self.advance()
        if word_str in typeValues:
            

            return Token(TT_TYPEKEYWORD, word_str, pos_start, self.pos)
        elif word_str in ("True", "False"):
            return Token(TT_BOOL, word_str, pos_start, self.pos)
        elif word_str in keywords:
            return Token(TT_KEYWORD, word_str, pos_start, self.pos)
        elif word_str in (TT_XOR, TT_AND, TT_OR, TT_NOT):
            return Token(word_str, pos_start=self.pos)
        else: 
            return Token(TT_IDENTIFIER, word_str, pos_start, self.pos)

    def make_literal(self):
        word_str = ""
        pos_start = self.pos.copy()
        self.advance()
        while self.current_char!='"':
            word_str+=self.current_char
            self.advance()
        self.advance()
        return Token(TT_LITERAL, word_str, pos_start, self.pos)
    def make_number(self):
        num_str = ""
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in DIGITS + ".":
            if self.current_char == ".":
                if dot_count == 1:
                    break
                dot_count += 1
                num_str += "."
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)


#######################################
# NODES
#######################################


class NumberNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    def __repr__(self):
        return f"{self.tok}"

class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end

class VarAssignNode:
    def __init__(self, var_name_tok, value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.value_node.pos_end
class VarReassignNode:
    def __init__(self, var_name_tok, value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.value_node.pos_end
class CharNode:
    def __init__(self, tok):
        self.tok  = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    def __repr__(self):
        return f"{self.tok}"
class BooleanNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    def __repr__(self):
        return f"{self.tok}"
class StringNode:
    def __init__(self, tok):
        self.tok  = tok
        
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    def __repr__(self):
        return f"{self.tok}"
class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node
        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end
    def __repr__(self):
        return f"({self.left_node}, {self.op_tok}, {self.right_node})"


class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node
        self.pos_start = self.op_tok.pos_start
        self.pos_end = self.op_tok.pos_end

    def __repr__(self):
        return f"({self.op_tok}, {self.node})"

class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case
        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (self.else_case or self.cases[len(self.cases)-1][0]).pos_end

class ForNode:
    def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node):
        self.var_name_tok = var_name_tok
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.body_node = body_node

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.obdy_node.pos_end


    class WhileNode:
        def __init__(self, conditoin_node, body_node):
            self.condition_node = condition_node
            self.body_node = body_node
            self.pos_start = self.condition_node.pos_start
            self.pos_end = self.body_node.pos_end


#######################################
# PARSE RESULT
#######################################


class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error:
                self.error = res.error
            return res.node

        return res

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        self.error = error
        return self


#######################################
# PARSER
#######################################


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(
        self,
    ):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok
    def unadvance(self):
        self.tok_idx-=1
        self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    def parse(self):
        res = self.boolExpr()
        if not res.error and self.current_tok.type != TT_EOF:
            #print(self.current_tok)
            return res.failure(
                InvalidSyntaxError(
                    self.current_tok.pos_start,
                    self.current_tok.pos_end,
                    "Expected '+', '-', '*' or '/'",
                )
            )
        return res

    ###################################
    def atom(self):
        res = ParseResult()

        tok = self.current_tok
        if tok.type in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))
        elif tok.type == TT_IDENTIFIER:
           
            res.register(self.advance())
            return res.success(VarAccessNode(tok))
        elif tok.type == TT_BOOL:
            res.register(self.advance())
            return res.success(BooleanNode(tok))
        elif tok.type == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error:
                return res
            if self.current_tok.type == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(
                    InvalidSyntaxError(
                        self.current_tok.pos_start,
                        self.current_tok.pos_end,
                        "Expected ')'",
                    )
                )
        elif tok.matches(TT_KEYWORD, "if"):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)
        elif tok.matches(TT_KEYWORD, "for"):
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)
        elif tok.matches(TT_KEYWORD, "while"):
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)
        return res.failure(InvalidSyntaxError(tok.pos_start, tok.pos_end, "Expected int, float, '+','-',or '('"))
    def for_expr(self):

        res = parseResult()
        res.register(self.advance())

        if self.current_tok.type != TT_TYPEKEYWORD:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected type keyword e.g. 'int'"))
        res.register(self.advance())
        if self.current_tok.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected identifier"))
        var_name = self.current_tok
        res.register(self.advance())
        if self.current_tok.type != TT_ASSIGNMENT:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, f"Expected '=' found {current_tok}"))
        res.register(self.advance())
        start_value = res.register(self.expr())
        if res.error: return res
        if self.current_tok.type !=TT_SEMICOLON:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ';' after expression"))
        res.register(self.advance())

        stop_condition = res.register(self.boolExpr())
        if res.error: return res

        if self.current_tok.type != TT_SEMICOLON:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ';' after expression"))
        
        res.register(self.advance())



        



    def if_expr(self):
        res = ParseResult()
        cases = []
        else_case = None
        
        res.register(self.advance())
        if not self.current_tok.type == TT_LPAREN:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected '('"))
        res.register(self.advance())
        condition = res.register(self.boolExpr())
        if res.error: return res
        if not self.current_tok.type == TT_RPAREN:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected ')'"))
        res.register(self.advance())
        expr = res.register(self.boolExpr())
        if res.error: return res
        cases.append((condition, expr))
        while self.current_tok.matches(TT_KEYWORD, 'elif'):
            res.register(self.advance())
            if not self.current_tok.type == TT_LPAREN:
                return res.failure(InvalidSyntaxError(tok.pos_start, tok.pos_end, "Expected '('"))
            res.register(self.advance())
            condition = res.register(self.boolExpr())
            if res.error: return res
            if not self.current_tok.type == TT_RPAREN:
                return res.failure(InvalidSyntaxError(tok.pos_start, tok.pos_end, "Expected ')'"))
            res.register(self.advance())
            expr = res.register(self.boolExpr())
            if res.error: return res
            cases.append((condition, expr))
        if self.current_tok.matches(TT_KEYWORD, "else"):
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            else_case = expr
        return res.success(IfNode(cases, else_case))


            

                


        

    def power(self):
        return self.bin_op(self.atom, (TT_POWER), self.factor)
    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error:
                return res
            return res.success(UnaryOpNode(tok, factor))
        return self.power()

        

        
    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV))
    def boolExpr(self):
        res = ParseResult()

        

        return self.bin_op(self.expr, [TT_COMPARE, TT_GREATERTHAN, TT_GREATERTHANEQ, TT_LESSTHAN, TT_LESSTHANEQ, TT_NOTEQUALTO, TT_NOT, TT_XOR, TT_AND, TT_OR])
    def expr(self):
        res = ParseResult()
       

        if self.current_tok.type == TT_CHAR:
            tok = self.current_tok
            res.register(self.advance())
            return res.success(CharNode(tok))
        if self.current_tok.type == TT_LITERAL:
            tok = self.current_tok
            res.register(self.advance())
            return res.success(StringNode(tok))  
        if self.current_tok.type == TT_IDENTIFIER:
            print("yes")
            var_name = self.current_tok
            res.register(self.advance())
            if self.current_tok.type != TT_ASSIGNMENT:
                res.register(self.unadvance())
            else:
                res.register(self.advance())
                expr = res.register(self.expr())
                if res.error: return result
                return res.success(VarReassignNode(var_name, expr))
        if self.current_tok.matchesList(TT_TYPEKEYWORD, typeValues):
            var_type = self.current_tok
            res.register(self.advance())

                
            if(self.current_tok.type != TT_IDENTIFIER):
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end,"Expected Identifier"))
            #identifier name or variable name
            var_name = self.current_tok
            res.register(self.advance())
            if self.current_tok.type != TT_ASSIGNMENT:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, "Expected assignment"))
            res.register(self.advance())
            if self.current_tok.type == TT_LITERAL and var_type.value != "string":
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, f"Expected type '{var_type.value}' got type 'string'"))
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))



        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    ###################################

    def bin_op(self, func, ops, func_b=None):
        if(func_b == None):
            func_b = func
        res = ParseResult()
        left = res.register(func())
        if res.error:
            return res

        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func_b())
            if res.error:
                return res
            left = BinOpNode(left, op_tok, right)

        return res.success(left)




def string_with_arrows(text, pos_start, pos_end):
    result = ''

    # Calculate indices
    idx_start = max(text.rfind('\n', 0, pos_start.idx), 0)
    idx_end = text.find('\n', idx_start + 1)
    if idx_end < 0: idx_end = len(text)
    
    # Generate each line
    line_count = pos_end.ln - pos_start.ln + 1
    for i in range(line_count):
        # Calculate line columns
        line = text[idx_start:idx_end]
        col_start = pos_start.col if i == 0 else 0
        col_end = pos_end.col if i == line_count - 1 else len(line) - 1

        # Append to result
        result += line + '\n'
        result += ' ' * col_start + '^' * (col_end - col_start)

        # Re-calculate indices
        idx_start = idx_end
        idx_end = text.find('\n', idx_start + 1)
        if idx_end < 0: idx_end = len(text)

    return result.replace('\t', '')

#runtime result class to keep track of the runtime result

class RunTimeResult:
    def __init__(self):
        self.value = None
        self.error = None
    def register(self, res):
        if res.error: self.error = res.error
        return res.value
    def success(self, value):
        self.value = value
        return self
    def failure(self, error):
        self.error = error
        return self



 #interpreter

class Number:
    def __init__(self, value):

        self.value = value
        self.set_pos()
        self.set_context()
    def set_context(self, context=None):
        self.context = context
        return self
    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
    def added_to(self, other):
        if isinstance(other, Number):
            num = Number(self.value+other.value)
            num.set_context(self.context)
            return num, None
    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value-other.value).set_context(self.context), None
    def mult_by(self, other):
        if isinstance(other, Number):
            return Number(self.value*other.value).set_context(self.context), None
    def div_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RunTimeError(other.pos_start, other.pos_end, "Division by Zero", self.context)
            return Number(self.value/other.value).set_context(self.context), None
    def powed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value**other.value).set_context(self.context), None
    def equal_to(self, other):
        if isinstance(other, Number):
            return Boolean(self.value == other.value).set_context(self.context), None
    def greater_than(self, other):
        if isinstance(other, Number):
            return Boolean(self.value > other.value).set_context(self.context), None
    def greater_than_eq(self, other):
        if isinstance(other, Number):
            return Boolean(self.value >= other.value).set_context(self.context), None
    def less_than(self, other):
        if isinstance(other, Number):
            return Boolean(self.value < other.value).set_context(self.context), None
    def less_than_eq(self, other):
        if isinstance(other, Number):
            return Boolean(self.value <= other.value).set_context(self.context), None
    def not_eq(self, other):
        if isinstance(other, Number):
            return Boolean(self.value!=other.value).set_context(self.context), None

    def is_true(self):
        return self.value !=0
    def __repr__(self):
        return str(self.value)

class Boolean:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()

    def set_context(self, context=None):
        self.context = context 
        return self
    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end 
    def is_true(self):
        return self.value =="True" or self.value ==True
    def equal_to(self, other):
        if isinstance(other, Boolean):
            return Boolean(self.value == other.value).set_context(self.context), None
    def is_and(self, other):
        self.fix_values(other)

        if isinstance(other, Boolean):

            return Boolean(self.value and other.value).set_context(self.context), None
    def is_or(self, other):
        if isinstance(other, Boolean):
            self.fix_values(other)
            
            return Boolean(self.value or other.value).set_context(self.context), None
    def fix_values(self, other):
        if other.value == "False": other.value = False
        if other.value == "True" : other.value = True
        if self.value == "True": self.value = True
        if self.value == "False": self.value = False
    def is_xor(self, other):
        self.fix_values(other)

        if isinstance(other, Boolean):
            return Boolean(self.value!=other.value).set_context(self.context), None
    def __repr__(self):
        return str(self.value)

class String:
    def __init__(self, value):

        self.value = value
        self.set_pos()
        self.set_context()

    def set_context(self, context=None):
        self.context = context
        return self
    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
    def __repr__(self):
        return str(self.value)

class Char:
    def __init__(self, value):

        self.value = value
        self.set_pos()
        self.set_context()

    def set_context(self, context=None):
        self.context = context
        return self
    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
    def __repr__(self):
        return str(self.value)
class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None





#symbols

class SymbolTable:
    def __init__(self):
        #storing as hashmap
        self.symbols = {}
        #parent symbol table for when we have other functions. and globals
        self.parent = None
    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value
    def remove(self, name):
        print(self.symbols)

        del self.symbols[name]
        print(self.symbols)
        print("remove from symbol table")






#interpreter

class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):

        raise Exception(f'No visit_{type(node).__name__}')
    
    #need a visit method for each type of thing in syntax tree
   
    def visit_BooleanNode(self, node, context):
        boolean = Boolean(node.tok.value)
        boolean.set_pos(node.pos_start, node.pos_end)
        boolean.set_context(context)
        return RunTimeResult().success(boolean)
    def visit_VarAccessNode(self, node, context):
        res = RunTimeResult()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)
        #if the value is not in the symbol table
        if not value: 
            return res.failure(RunTimeError(node.pos_start, node.pos_end, f"'{var_name}' is not defined", context))

        return res.success(value)
    def visit_CharNode(self, node, context):
        char = Char(node.tok.value)
        char.set_pos(node.pos_start, node.pos_end)
        char.set_context(context)
        return RunTimeResult().success(char)
    def visit_StringNode(self, node, context):
        literal = String(node.tok.value)
        literal.set_pos(node.pos_start, node.pos_end)
        literal.set_context(context)
        return RunTimeResult().success(literal)
    def visit_VarReassignNode(self, node, context):
        res = RunTimeResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, context))
        if res.error: return res
        value = context.symbol_table.get(var_name)
        if not value:
            return res.failure(RunTimeError(node.pos_start, node.pos_end, f"'{var_name}' must be declared first", context))
        context.symbol_table.remove(var_name)
        context.symbol_table.set(var_name, value)
        return RunTimeResult().success(value)

    def visit_VarAssignNode(self, node, context):
        res = RunTimeResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, context))
        if res.error: return res

        context.symbol_table.set(var_name, value)
        return res.success(value)

    def visit_NumberNode(self, node, context):

        number = Number(node.tok.value)
        number.set_pos(node.pos_start, node.pos_end)
        number.set_context(context)
        return RunTimeResult().success(number)
    def visit_BinOpNode(self, node, context):
        res = RunTimeResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error: return res
        right = res.register(self.visit(node.right_node, context))
        if res.error: return res
        
        if node.op_tok.type == TT_PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.type == TT_COMPARE:
            result, error = left.equal_to(right)
        elif node.op_tok.type == TT_MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == TT_MUL:
            result, error = left.mult_by(right)
        elif node.op_tok.type == TT_DIV:
            result, error = left.div_by(right)
        elif node.op_tok.type == TT_POWER:
            result, error =  left.powed_by(right)
        elif node.op_tok.type == TT_GREATERTHAN:
            result, error = left.greater_than(right)

        elif node.op_tok.type == TT_GREATERTHANEQ:
            result, error = left.greater_than_eq(right)
        elif node.op_tok.type == TT_LESSTHAN:
            result, error = left.less_than(right)
        elif node.op_tok.type == TT_LESSTHANEQ:
            result, error = left.less_than_eq(right)
        elif node.op_tok.type == TT_NOTEQUALTO:
            result, error = left.not_eq(right)
        elif node.op_tok.type == TT_AND:
            result, error = left.is_and(right)
        elif node.op_tok.type == TT_OR:

            result, error = left.is_or(right)
        elif node.op_tok.type == TT_XOR:
            result, error = left.is_xor(right)
        
        if error:
            return res.failure(error)
        else:
            result.set_pos(node.pos_start, node.pos_end)
        
            return res.success(result)
        

        self.visit(node.left_node)
        self.visit(node.right_node)
    def visit_UnaryOpNode(self, node, context):
        res = RunTimeResult()
        number = res.register(self.visit(node.node, context))
        if res.error:
            return res
        error = None
        if node.op_tok.type == TT_MINUS:
            number = number.mult_by(Number(-1))
        if error:
            return res.failure(error)
        else:

            return res.success(number.set_pos(node.pos_start, node.pos_end))
    def visit_IfNode(self, node, context):
        res =RunTimeResult()

        for condition, expr in node.cases:
            condition_value =res.register(self.visit(condition, context))
            if res.error: return res

            if condition_value.is_true():
                
                expr_value = res.register(self.visit(expr, context))
                if res.error: return res
                return res.success(expr_value)
        if node.else_case:
            else_value = res.register(self.visit(node.else_case, context))
            if res.error: return result
            return res.success(else_value)
        return res.success(None)


#######################################
# RUN
#######################################

global_symbol_table = SymbolTable()
global_symbol_table.set("null", Number(0))
def run(fn, text):
    # Generate tokens
    lexer = Lexer(fn, text)
    
    tokens, error = lexer.make_tokens()
    print(tokens)
    if error:
        return None, error

    # Generate AST
    parser = Parser(tokens)
    ast = parser.parse()
    
    if ast.error: return None, ast.error

    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table

    result = interpreter.visit(ast.node, context)

    
    return result.value, result.error
