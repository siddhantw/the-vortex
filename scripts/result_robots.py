#!/usr/bin/env python

"""Usage: gather_failed_tests.py inpath [outpath]

Reads result of a test run from Robot output file and gathers failed
test names to an argument file (default failed_tests.txt). 

To re-run failed tests:
pybot --argumentfile failed_tests.txt ..
"""

import sys

from robot.api import ExecutionResult


def number_of_failed_tests(outputfile):
    result = ExecutionResult(outputfile)
    result.configure(stat_config={'suite_stat_level': 2,
                                  'tag_stat_combine': 'tagANDanother'})
    stats = result.statistics

    print(stats.total.critical.failed)
    # print "in python function"
    # print stats.total.critical.passed

    return (stats.total.critical.failed)


if __name__ == '__main__':
    try:
        outpath = number_of_failed_tests(*sys.argv[1:])

    except TypeError:
        print(__doc__)
