#!/usr/bin/env python3
"""
Quick Azure OpenAI Test with Your Configuration
Tests with the specific endpoint and deployment you provided
"""

import os
import sys
import json
from datetime import datetime

# Add the scripts directory to the path
sys.path.append('')

def quick_test_with_your_config():
    """Test Azure OpenAI with your specific configuration"""

    print("=" * 60)
    print("Azure OpenAI Test - Your Configuration")
    print("=" * 60)

    # Your specific configuration
    config = {
        "endpoint": "https://mh-open-ai-east-us2.openai.azure.com/",
        "deployment": "dev-chat-ai-gpt4.1-mini",
        "api_version": "2024-10-21"
    }

    print(f"🎯 Testing with your configuration:")
    print(f"   Endpoint: {config['endpoint']}")
    print(f"   Deployment: {config['deployment']}")
    print(f"   API Version: {config['api_version']}")

    # Check for API key
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not api_key or api_key == "your-actual-api-key-here":
        print("\n❌ API Key not provided or still placeholder")
        print("\nTo test Azure OpenAI, please:")
        print("1. Get your actual API key from Azure Portal")
        print('2. Run: export AZURE_OPENAI_API_KEY="your-actual-api-key-here"')
        print("3. Run this script again")
        return False

    try:
        from azure_openai_client import AzureOpenAIClient

        print(f"\n🔑 API Key: {'*' * 20} (found)")
        print("\n🚀 Testing connection...")

        # Initialize client with your configuration
        client = AzureOpenAIClient(
            azure_endpoint=config['endpoint'],
            api_key=api_key,
            deployment_name=config['deployment'],
            api_version=config['api_version']
        )

        # Quick test prompts
        test_prompts = [
            {
                "name": "Connection Test",
                "content": "Say 'Hello from Azure OpenAI!' to confirm the connection is working.",
                "max_tokens": 20
            },
            {
                "name": "Math Check",
                "content": "What is 25 + 17? Just give me the number.",
                "max_tokens": 10
            },
            {
                "name": "Code Understanding",
                "content": "In one sentence, what does this Python code do: def greet(name): return f'Hello, {name}!'",
                "max_tokens": 50
            },
            {
                "name": "Test Generation",
                "content": "Write a simple test case title for testing a user login feature.",
                "max_tokens": 30
            }
        ]

        print(f"\n📝 Running {len(test_prompts)} test prompts...")

        results = []
        total_tokens = 0

        for i, test in enumerate(test_prompts, 1):
            print(f"\n{i}. {test['name']}")
            print(f"   Prompt: {test['content'][:60]}...")

            try:
                start_time = datetime.now()

                response = client.chat_completion_create(
                    model=config['deployment'],
                    messages=[{"role": "user", "content": test['content']}],
                    max_tokens=test['max_tokens'],
                    temperature=0.7
                )

                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()

                if response and response.get('choices'):
                    content = response['choices'][0]['message']['content'].strip()
                    usage = response.get('usage', {})
                    tokens_used = usage.get('total_tokens', 0)
                    total_tokens += tokens_used

                    print(f"   ✅ Response ({response_time:.2f}s, {tokens_used} tokens):")
                    print(f"   📄 {content}")

                    results.append({
                        "test": test['name'],
                        "status": "success",
                        "response": content,
                        "response_time": response_time,
                        "tokens": tokens_used
                    })

                else:
                    print("   ❌ No response received")
                    results.append({"test": test['name'], "status": "no_response"})

            except Exception as e:
                print(f"   ❌ Error: {str(e)[:100]}...")
                results.append({"test": test['name'], "status": "error", "error": str(e)})

        # Results summary
        print("\n" + "=" * 60)
        print("AZURE OPENAI TEST RESULTS")
        print("=" * 60)

        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] != 'success']

        print(f"✅ Successful tests: {len(successful)}/{len(results)}")
        print(f"❌ Failed tests: {len(failed)}")

        if successful:
            avg_time = sum(r['response_time'] for r in successful) / len(successful)
            print(f"📊 Total tokens consumed: {total_tokens}")
            print(f"⏱️  Average response time: {avg_time:.2f} seconds")

            print(f"\n🎉 Azure OpenAI is working correctly with your configuration!")
            print(f"   Endpoint: {config['endpoint']}")
            print(f"   Deployment: {config['deployment']}")

            # Save test results
            result_data = {
                "timestamp": datetime.now().isoformat(),
                "configuration": config,
                "results": results,
                "summary": {
                    "successful_tests": len(successful),
                    "total_tests": len(results),
                    "total_tokens": total_tokens,
                    "average_response_time": avg_time
                }
            }

            with open('../../azure_openai_test_results.json', 'w') as f:
                json.dump(result_data, f, indent=2)

            print(f"📄 Detailed results saved to: azure_openai_test_results.json")
            return True

        else:
            print(f"\n💥 All tests failed. Check your configuration and API key.")
            if failed:
                print("Errors encountered:")
                for fail in failed:
                    if 'error' in fail:
                        print(f"  - {fail['test']}: {fail['error'][:100]}...")
            return False

    except ImportError as e:
        print(f"❌ Error importing Azure OpenAI client: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Azure OpenAI Quick Test")
    success = quick_test_with_your_config()

    if success:
        print("\n🎊 Test completed successfully! Azure OpenAI is ready to use.")
    else:
        print("\n🔧 Please check your API key and try again.")

    sys.exit(0 if success else 1)
