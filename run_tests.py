#!/usr/bin/env python
"""
Test runner script with coverage reporting.
"""

import os
import sys
import pytest
import coverage
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_tests():
    """Run all tests with coverage reporting."""
    # Start coverage
    cov = coverage.Coverage(
        branch=True,
        source=['.'],
        omit=[
            'venv/*',
            'tests/*',
            '*/__pycache__/*',
            '*/migrations/*',
            'run_tests.py'
        ]
    )
    cov.start()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = f"test_reports/{timestamp}"
    
    # Create report directory
    os.makedirs(report_dir, exist_ok=True)
    
    # Run tests
    args = [
        '--verbose',
        '--cov=.',
        f'--cov-report=html:{report_dir}/coverage',
        f'--cov-report=xml:{report_dir}/coverage.xml',
        f'--junit-xml={report_dir}/junit.xml',
        '--durations=10',
        '--maxfail=10',
        '-p', 'no:warnings'
    ]
    
    # Add test categories if specified
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == 'unit':
            args.append('tests/unit')
        elif test_type == 'integration':
            args.append('tests/integration')
        elif test_type == 'e2e':
            args.append('tests/e2e')
        else:
            args.append('tests')
    else:
        args.append('tests')
    
    # Run pytest
    result = pytest.main(args)
    
    # Stop coverage
    cov.stop()
    cov.save()
    
    # Generate coverage reports
    logger.info("Generating coverage reports...")
    cov.html_report(directory=f"{report_dir}/coverage")
    cov.xml_report(outfile=f"{report_dir}/coverage.xml")
    
    # Print summary
    logger.info("\nTest Summary:")
    logger.info(f"- Report directory: {report_dir}")
    logger.info(f"- Coverage report: {report_dir}/coverage/index.html")
    logger.info(f"- JUnit report: {report_dir}/junit.xml")
    
    return result

if __name__ == '__main__':
    sys.exit(run_tests()) 