from transformers import pipeline

summarizer = pipeline('summarization')

text = """Hello, hello! Hi Kashi, thanks for meeting with me. How can we assist your hiring needs?

Hi, we need help finding a couple of software engineers.

Got it. What skills are you looking for in these engineers?

They should have experience in Python and machine learning.

That's great. Do you need this position filled within the next two months?

Yes.

We will start searching for candidates right away and keep you updated.

Thanks, I appreciate it. We will be in touch soon.
"""

summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

print(summary)
