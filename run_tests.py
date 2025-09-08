import sys, unittest
if __name__ == "__main__":
    # default to verbose discovery in tests/
    argv = ["discover", "-s", "tests", "-p", "test*.py", "-v"]
    unittest.main(module=None, argv=[""] + argv)
