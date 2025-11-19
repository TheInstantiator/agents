#!/usr/bin/env python3
"""
Test that llm_overrides parameter works correctly in MyStockPicker crew.

This test verifies that:
1. Default LLM assignments are correct
2. Overrides are applied correctly
3. Fixed agents (finders) ignore overrides
4. Configurable agents (manager, researcher, rater) respect overrides
"""
import sys
from pathlib import Path

# Add paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(Path(__file__).parent / "src"))

from my_stock_picker.crew import MyStockPicker


def test_default_llm_assignments():
    """Test that default LLM assignments are correct"""
    print("=" * 60)
    print("Test 1: Default LLM Assignments")
    print("=" * 60)

    crew = MyStockPicker()

    # Check defaults
    assert hasattr(crew, 'llm_grok'), "Should have llm_grok"
    assert hasattr(crew, 'llm_gemini'), "Should have llm_gemini"
    assert hasattr(crew, 'llm_ollama'), "Should have llm_ollama"

    # Check role assignments exist
    assert hasattr(crew, 'manager_llm'), "Should have manager_llm"
    assert hasattr(crew, 'researcher_llm'), "Should have researcher_llm"
    assert hasattr(crew, 'rater_llm'), "Should have rater_llm"

    # Check defaults are correct (should match gemini/grok as per docstring)
    # manager: gemini-2.5-flash, researcher: grok-4-fast, rater: grok-4-fast
    assert crew.manager_llm == crew.llm_gemini, "Default manager should be Gemini"
    assert crew.researcher_llm == crew.llm_grok, "Default researcher should be Grok"
    assert crew.rater_llm == crew.llm_grok, "Default rater should be Grok"

    print("✓ Default assignments correct:")
    print(f"  - manager_llm: gemini-2.5-flash")
    print(f"  - researcher_llm: grok-4-fast")
    print(f"  - rater_llm: grok-4-fast")
    print()


def test_override_all_to_ollama():
    """Test overriding all configurable agents to use Ollama"""
    print("=" * 60)
    print("Test 2: Override All to Ollama")
    print("=" * 60)

    crew = MyStockPicker(llm_overrides={
        'manager': 'ollama-gemma27b',
        'researcher': 'ollama-gemma27b',
        'rater': 'ollama-gemma27b'
    })

    assert crew.manager_llm == crew.llm_ollama, "Manager should be overridden to Ollama"
    assert crew.researcher_llm == crew.llm_ollama, "Researcher should be overridden to Ollama"
    assert crew.rater_llm == crew.llm_ollama, "Rater should be overridden to Ollama"

    print("✓ All agents overridden to Ollama:")
    print(f"  - manager_llm: ollama-gemma27b")
    print(f"  - researcher_llm: ollama-gemma27b")
    print(f"  - rater_llm: ollama-gemma27b")
    print()


def test_partial_override():
    """Test partial override (only some agents)"""
    print("=" * 60)
    print("Test 3: Partial Override")
    print("=" * 60)

    crew = MyStockPicker(llm_overrides={
        'researcher': 'gemini-2.5-flash'  # Only override researcher
    })

    # Manager and rater should remain default
    assert crew.manager_llm == crew.llm_gemini, "Manager should remain default (Gemini)"
    assert crew.rater_llm == crew.llm_grok, "Rater should remain default (Grok)"

    # Researcher should be overridden
    assert crew.researcher_llm == crew.llm_gemini, "Researcher should be overridden to Gemini"

    print("✓ Partial override works:")
    print(f"  - manager_llm: gemini-2.5-flash (default)")
    print(f"  - researcher_llm: gemini-2.5-flash (overridden)")
    print(f"  - rater_llm: grok-4-fast (default)")
    print()


def test_agents_use_correct_llms():
    """Test that agent definitions use the correct LLM attributes"""
    print("=" * 60)
    print("Test 4: Agent Definitions Use Correct LLMs")
    print("=" * 60)

    crew = MyStockPicker(llm_overrides={
        'manager': 'ollama-gemma27b',
        'researcher': 'gemini-2.5-flash',
        'rater': 'ollama-gemma27b'
    })

    # Get agent instances
    manager_agent = crew.crew().manager_agent
    researcher_agent = crew.financial_researcher()
    rater_agent = crew.stock_rater()

    # Verify agents use the configured LLMs
    assert manager_agent.llm == crew.llm_ollama, "Manager agent should use Ollama"
    assert researcher_agent.llm == crew.llm_gemini, "Researcher agent should use Gemini"
    assert rater_agent.llm == crew.llm_ollama, "Rater agent should use Ollama"

    print("✓ Agents correctly use configured LLMs:")
    print(f"  - manager agent: ollama-gemma27b")
    print(f"  - researcher agent: gemini-2.5-flash")
    print(f"  - rater agent: ollama-gemma27b")
    print()


def test_fixed_agents_ignore_overrides():
    """Test that fixed agents (finders) always use their designated LLMs"""
    print("=" * 60)
    print("Test 5: Fixed Agents Ignore Overrides")
    print("=" * 60)

    # Even with overrides, finders should use their hardcoded LLMs
    crew = MyStockPicker(llm_overrides={
        'manager': 'ollama-gemma27b',
        'researcher': 'ollama-gemma27b',
        'rater': 'ollama-gemma27b'
    })

    finder_grok = crew.trending_company_finder_grok()
    finder_gemini = crew.trending_company_finder_gemini()

    # These should always be hardcoded regardless of overrides
    assert finder_grok.llm == crew.llm_grok, "Grok finder should always use Grok"
    assert finder_gemini.llm == crew.llm_gemini, "Gemini finder should always use Gemini"

    print("✓ Fixed agents maintain hardcoded LLMs:")
    print(f"  - trending_company_finder_grok: grok-4-fast (hardcoded)")
    print(f"  - trending_company_finder_gemini: gemini-2.5-flash (hardcoded)")
    print()


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Testing LLM Override Functionality")
    print("=" * 60 + "\n")

    tests = [
        test_default_llm_assignments,
        test_override_all_to_ollama,
        test_partial_override,
        test_agents_use_correct_llms,
        test_fixed_agents_ignore_overrides
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test failed: {test_func.__name__}")
            print(f"  Error: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ Test crashed: {test_func.__name__}")
            print(f"  Error: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
