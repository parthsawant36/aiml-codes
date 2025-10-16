import random

# Simple AI responses
ai_responses = [
    "I enjoy reading books in my free time.",
    "The weather is pleasant today.",
    "Pizza is my favorite food because it tastes great.",
    "I would love to fly if I had a superpower."
]

while True:
    question = input("Interrogator: ")
    if question.lower() == "exit":
        break
    print("Machine:", random.choice(ai_responses))


