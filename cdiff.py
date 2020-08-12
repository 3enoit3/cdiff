#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Diff"""

import sys
import argparse
import unittest
import logging
import subprocess
import itertools

# Words
def is_word(c):
    """Test if a character belongs to a word"""
    return str(c).isalnum()

def is_breaking_word(s1, s2):
    """Test if the end of s1 and the beginning of s2 belongs to a word"""
    return s1 and s2 and is_word(s1[-1]) and is_word(s2[0])

# Fragments
def explode(s, pos, length):
    """Extract a sub-string, and before and after substrings"""
    return s[0:pos], s[pos:pos+length], s[pos+length:]

def get_fragments(s, length):
    """Generates all substrings of fixed length"""
    count = len(s) - length + 1
    for i in range(count):
        _, fragment, _ = explode(s, i, length)
        yield i, fragment

def get_all_fragments(s, smallest=3):
    """Generates all substrings from the biggest to the smallest"""
    for i in range(len(s), smallest-1, -1):
        yield i, get_fragments(s, i)

def get_common_candidates(s1, s2, smallest=3):
    """Generates all common substrings from the biggest to the smallest"""
    small, big = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    for length, fragments in get_all_fragments(small, smallest):
        for pos, fragment in fragments:
            found = big.find(fragment)
            while found != -1:
                small_fields = explode(small, pos, length)
                big_fields = explode(big, found, length)
                yield zip(small_fields, big_fields) if len(s1) <= len(s2) else zip(big_fields, small_fields)
                found = big.find(fragment, found + 1)
    yield [("",""), (small,big), ("","")] if len(s1) <= len(s2) else [("",""), (big,small), ("","")]

# Diffs
# A diff is a list of pairs [(s1_1, s2_1), (s1_2, s2_2)...].
# The common parts are pairs of the same string (s, s).

# Normalize
def clean_head(iterable):
    """Remove leading empty pairs"""
    empty = lambda x: not x[0] and not x[1]
    for x in itertools.dropwhile(empty, iterable):
        yield x

def normalize(diff):
    """Reduce the diff to the smallest form (no empty pairs, no repetitions)"""
    diff = list(clean_head(diff))
    if not diff:
        return diff

    normal = [diff[0]]
    for d1, d2 in diff[1:]:
        if d1 or d2:
            m1, m2 = normal[-1]
            if d1 == d2 and m1 == m2:
                normal[-1] = (m1 + d1, m2 + d2)
            elif d1 != d2 and m1 != m2:
                normal[-1] = (m1 + d1, m2 + d2)
            else:
                normal.append((d1, d2))
    return normal

# Improve
def gen_left(s):
    """Generate all splits for s, from right most to left"""
    if s:
        for i in range(len(s)-1, 0, -1):
            yield (s[:i], s[i:])

def gen_right(s):
    """Generate all splits for s, from left most to right"""
    for i in range(1, len(s)):
        yield (s[:i], s[i:])

def expand_diff(lhs, rhs):
    """Move the boundary between common and diff fragments so that diff contains whole words"""
    l1, l2 = lhs
    r1, r2 = rhs
    if is_breaking_word(l1, r1) and is_breaking_word(l2, r2):
        if l1 == l2:
            for before, after in gen_left(l1):
                if not is_breaking_word(before, after):
                    return (before, before), (after + r1, after + r2)
            return ("", ""), (l1 + r1, l2 + r2)
        if r1 == r2:
            for before, after in gen_right(r1):
                if not is_breaking_word(before, after):
                    return (l1 + before, l2 + before), (after, after)
            return (l1 + r1, l2 + r2), ("", "")

    return lhs, rhs

def improve_diff(diff):
    """Make diff more readable for humans"""
    if len(diff) <= 1:
        return diff

    new_diff = []
    prev = diff[0]
    for curr in diff[1:]:
        ready, prev = expand_diff(prev, curr)
        new_diff.append(ready)
    new_diff.append(prev)

    return normalize(new_diff)

def is_diff_nice_enough(diff):
    len_before = sum([len(s) for s, _ in diff])
    len_after = sum([len(s) for _, s in diff])
    len_common = sum([len(s1) for s1, s2 in diff if s1 == s2])
    len_diff = len([s1 for s1, s2 in diff if s1 != s2])
    return len_diff <= 3

# Computation
def get_broken_words(diff):
    """Count the number of broken words in a diff"""
    broken = 0

    old = [d for d, _ in diff]
    prev = old[0]
    for f in old[1:]:
        if is_breaking_word(prev, f):
            broken += 1
        prev = f

    new = [d for _, d in diff]
    prev = new[0]
    for f in new[1:]:
        if is_breaking_word(prev, f):
            broken += 1
        prev = f

    return broken

def get_diff_candidates(s1, s2, depth=1):
    """Generate diffs"""
    diffs = []
    def is_new(diff):
        if diff not in diffs:
            diffs.append(diff)
            return True

    for candidate in itertools.islice(get_common_candidates(s1, s2), depth):
        diff = []
        for common in candidate:
            if common == ("", ""):
                continue

            if common[0] == common[1]:
                diff += [common]
            else:
                if common == (s1, s2):
                    diff += [common]
                else:
                    deeper = list(itertools.islice(get_diff_candidates(*common), depth))[0]
                    diff += deeper
        if is_new(diff):
            yield diff

def evaluate(diff):
    if not diff:
        return 0

    # broken words
    broken = get_broken_words(diff)

    # smallest blocks
    # blocks = max(max([len(d) for d, _ in diff]), max([len(d) for _, d in diff]))
    blocks = 0

    # smallest fragments
    diff_fragments = [(d1, d2) for d1, d2 in diff if d1 != d2]
    if diff_fragments:
        small = sum([len(d1) + len(d2) for d1, d2 in diff_fragments if d1])
    else:
        small = 0

    # return broken + (999 - blocks * 100) + (small if small < 100 else 99) * 10000
    return broken + (small if small < 100 else 99) * 100

def weight_candidates(candidates, evaluate):
    weighted = [(evaluate(c), c) for c in candidates]
    return sorted(weighted, key=lambda x: x[0])

# Display
def align_str(s, length, align="^"):
    return ("{:" + align + str(length) + "}").format(s)

COLOR_RED = 91
COLOR_GREEN = 92
COLOR_YELLOW = 93
COLOR_BLUE = 94

def color_str(s, color):
    return '\033[{}m'.format(color) + str(s) + '\033[0m'

def format_diff(header, diffs):
    out = ""
    if header:
        out += str(header) + "\n"
    for d1, d2 in diffs:
        if d1 == d2:
            out += d1
        else:
            if d1:
                out += color_str("[", COLOR_BLUE)
            out += color_str(d1, COLOR_RED)
            if d1 and d2:
                out += color_str("⇒", COLOR_BLUE)
            out += color_str(d2, COLOR_GREEN)
            if d1:
                out += color_str("]", COLOR_BLUE)
    return out

# Api
def compute_diff(s1, s2, explain=False):
    if explain:
        print
        print color_str("input", COLOR_YELLOW) + ":"
        print "   '{}' -> '{}'".format(color_str(s1, COLOR_RED), color_str(s2, COLOR_GREEN))
        print

    if not s1 or not s2:
        return [(s1, s2)]

    candidates = itertools.islice(get_diff_candidates(s1, s2, 10), 10)
    ordered = weight_candidates(candidates, evaluate)
    _, diff = ordered[0]
    if explain:
        print color_str("candidates", COLOR_YELLOW) + ":"
        print " >", color_str(ordered[0], COLOR_GREEN)
        for candidate in ordered[1:40]:
            print "  ", candidate

    improved = improve_diff(diff)
    if improved != diff and explain:
        print
        print color_str("improved to", COLOR_YELLOW) + ":"
        print "  ", improved
        print

    if not is_diff_nice_enough(improved):
        if explain:
            print
            print color_str("diff is not nice enough", COLOR_RED)
            print
        return [(s1, s2)]

    return improved

# Diff tool
def decode_range(header_range):
    if "," in header_range:
        lines = header_range.split(",")
        return (int(lines[0]), int(lines[1]))
    else:
        line = int(header_range)
        return (line, line)

def decode_header(header):
    if "a" in header:
        ranges = header.split("a")
        return "add", decode_range(ranges[0]), decode_range(ranges[1])
    if "d" in header:
        ranges = header.split("d")
        return "delete", decode_range(ranges[0]), decode_range(ranges[1])
    if "c" in header:
        ranges = header.split("c")
        return "change", decode_range(ranges[0]), decode_range(ranges[1])
    return "none", None, None

def group_diffs(output):
    header = "none"
    s1, s2 = [], []
    for l in output.splitlines():
        if l.startswith("< "):
            s1.append(l[2:])
        elif l.startswith("> "):
            s2.append(l[2:])
        elif l != "---":
            if s1 or s2:
                yield header, "\n".join(s1), "\n".join(s2)
            header = l
            s1, s2 = [], []
    if s1 or s2:
        yield header, "\n".join(s1), "\n".join(s2)

def diff_files(f1, f2):
    try:
        subprocess.check_output(["diff", "-E", "-b", "-w", "-B", "-a", f1, f2], stderr=subprocess.STDOUT)
    except Exception as e:
        out = e.output

    for h, s1, s2 in group_diffs(out):
        header = decode_header(h)
        diff = compute_diff(s1, s2)
        print format_diff(header, diff)

def diff_files2(f1, f2):
    s1 = open(f1).read()
    s2 = open(f2).read()
    diff = compute_diff(s1, s2)
    print format_diff(None, diff)

# Main
def main():
    """Entry point"""

    # Parse options
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", default=False,
                        help="show debug information")
    parser.add_argument("f1", type=str, help="file before")
    parser.add_argument("f2", type=str, help="file after")
    args = parser.parse_args()

    # Configure debug
    if args.debug:
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
        logging.debug("Enabled debug logging")

    diff_files(args.f1, args.f2)

    return 0

if __name__ == "__main__":
    sys.exit(main())

class Tests(unittest.TestCase):
    """Unit tests"""
    # run test suite with "python -m unittest <this_module_name_without_py_extension>"

    def assert_diff(self, s1, s2, expected_diff, explain=False):
        diff = compute_diff(s1, s2, explain)
        if not explain and (diff != expected_diff):
            compute_diff(s1, s2, True) # recompute with explanations
        self.assertEqual(diff, expected_diff)

    def test_basics(self):
        """Characters"""

        same = lambda s: (s, s)
        add = lambda s: ("", s)
        sub = lambda s: (s, "")

        self.assert_diff("", "", [same("")])
        self.assert_diff("abcd", "abcd", [same("abcd")])
        self.assert_diff("abcd", "bcd", [sub("a"), same("bcd")])
        self.assert_diff("bcd", "abcd", [add("a"), same("bcd")])
        self.assert_diff("bcda", "bcd", [same("bcd"), sub("a")])
        self.assert_diff("bcd", "bcda", [same("bcd"), add("a")])
        self.assert_diff("aabcda", "abcd", [sub("a"), same("abcd"), sub("a")])
        self.assert_diff("abcd", "aabcda", [add("a"), same("abcd"), add("a")])
        self.assert_diff("ccaaaccbbbcc", "aaabbb", [sub("cc"), same("aaa"), sub("cc"), same("bbb"), sub("cc")])
        self.assert_diff("aaabbb", "ccaaaccbbbcc", [add("cc"), same("aaa"), add("cc"), same("bbb"), add("cc")])
        self.assert_diff("aaaccccbbbcccc", "cccee", [sub("aaaccccbbbc"), same("ccc"), add("ee")])
        self.assert_diff("cccee", "aaaccccbbbcccc", [add("aaaccccbbbc"), same("ccc"), sub("ee")])
        self.assert_diff("abcd", "efgh", [("abcd", "efgh")])

    def test_seen(self):
        """Real scenario"""

        same = lambda s: (s, s)
        add = lambda s: ("", s)
        sub = lambda s: (s, "")

        self.assert_diff("toto",
                         "this toto is nice",
                         [add("this "), same("toto"), add(" is nice")])
        self.assert_diff("this toto is nice",
                         "this toto is nicer",
                         [same("this toto is nice"), add("r")])
        self.assert_diff("    that toto is nice",
                         "    this toto is nice",
                         [same("    "), ("that", "this"), same(" toto is nice")])

        self.assert_diff("SRC_URI += \" \\",
                         "# use ${MACHINE} as a workaround, because normal way (SRC_URI_append_zc702-zynq7/-ax) seems to include both device-tree\nSRC_URI_append = \" \\",
                         [add("# use ${MACHINE} as a workaround, because normal way (SRC_URI_append_zc702-zynq7/-ax) seems to include both device-tree\n"), same("SRC_URI"), (" +", "_append "), same("= \" \\")])
        self.assert_diff("call /etc/network/ethernet", "call /etc/domino/ethernet", [same("call /etc/"), ("network", "domino"), same("/ethernet")])

        self.assert_diff('include_directories("${CMAKE_SOURCE_DIR}/BERT/include")\ninclude_directories("${CMAKE_SOURCE_DIR}/BERT/src")',
                         'set(BERT_SOURCES ${CMAKE_SOURCE_DIR}/BERT/src)\nset(BERT_INCLUDES ${CMAKE_SOURCE_DIR}/BERT/include)\ninclude_directories(BERT/src BERT/include)',
                         [('include_directories("${CMAKE_SOURCE_DIR}/BERT/include")\ninclude_directories("${CMAKE_SOURCE_DIR}/BERT/src")',
                         'set(BERT_SOURCES ${CMAKE_SOURCE_DIR}/BERT/src)\nset(BERT_INCLUDES ${CMAKE_SOURCE_DIR}/BERT/include)\ninclude_directories(BERT/src BERT/include)')])

    def test_improve(self):
        """Diff improvement"""

        # Left
        self.assertEqual(list(gen_left("")), [])
        self.assertEqual(list(gen_left("a")), [])
        self.assertEqual(list(gen_left("ab")), [("a","b")])
        self.assertEqual(list(gen_left("abcde")), [("abcd","e"), ("abc","de"), ("ab","cde"), ("a","bcde")])

        # Right
        self.assertEqual(list(gen_right("")), [])
        self.assertEqual(list(gen_right("a")), [])
        self.assertEqual(list(gen_right("ab")), [("a","b")])
        self.assertEqual(list(gen_right("abcde")), [("a","bcde"), ("ab","cde"), ("abc","de"), ("abcd","e")])

        # Extend
        self.assertEqual(expand_diff(("",""), ("","")), (("",""), ("","")))
        self.assertEqual(expand_diff(("ab","ab"), ("cd","de")), (("",""), ("abcd","abde")))
        self.assertEqual(expand_diff(("ab","cd"), ("de","de")), (("abde","cdde"), ("","")))

        # Improve
        self.assertEqual(improve_diff([]), [])
        self.assertEqual(improve_diff([("a","b")]), [("a","b")])
        self.assertEqual(improve_diff([("a","a"), ("b","c"), ("d","d")]), [("abd","acd")])

    def test_normalize(self):
        """Diff normalize"""

        # Clean leading empty fragment
        self.assertEqual(list(clean_head([])), [])
        self.assertEqual(list(clean_head([("","")])), [])
        self.assertEqual(list(clean_head([("",""), ("a","a")])), [("a","a")])
        self.assertEqual(list(clean_head([("a","a"), ("","")])), [("a","a"), ("","")])

        # Normalize
        self.assertEqual(normalize([]), [])
        self.assertEqual(normalize([("","")]), [])
        self.assertEqual(normalize([("",""), ("a","a")]), [("a","a")])
        self.assertEqual(normalize([("a","a"), ("","")]), [("a","a")])
        self.assertEqual(normalize([("a","b"), ("c","c")]), [("a","b"), ("c","c")])
        self.assertEqual(normalize([("a","a"), ("b","b")]), [("ab","ab")])
        self.assertEqual(normalize([("a","b"), ("c","d")]), [("ac","bd")])
