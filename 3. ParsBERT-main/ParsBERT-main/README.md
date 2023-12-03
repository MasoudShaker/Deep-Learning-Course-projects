






<!DOCTYPE html>
<html lang="en" data-color-mode="auto" data-light-theme="light" data-dark-theme="dark" data-a11y-animated-images="system">
  <head>
    <meta charset="utf-8">


# Multi-Layer Perceptron for Image Classification

Welome to the **third** programming assignment of the Deep Learning course. 

<p dir="auto"><strong>Please pay attention to these notes:</strong>
<br></p>
<ul dir="auto">
<li>If you need any additional information, please review the assignment page on the course github.</li>
<li>The items you need to answer are highlighted in red and the coding parts you need to implement are denoted by:</li>
</ul>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="########################################
#              Your Code               #
########################################"><pre class="notranslate"><code>########################################
#              Your Code               #
########################################
</code></pre></div>
<ul dir="auto">
<li>Finding any sort of copying will zero down that assignment grade.</li>
<li>If you have any questions about this assignment, feel free to ask us.</li>
<li>You must run this notebook on Google Colab platform, it depends on Google Colab VM for some of the depencecies.</li>
</ul>
<p dir="auto"><br><br></p>
<h2 dir="auto"><a id="user-content-bert" class="anchor" aria-hidden="true" href="#bert"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>BERT</h2>
<p dir="auto">BERT stands for Bi-directional Encoder Representation from Transformers is designed to pre-train deep bidirectional representations from unlabeled texts by jointly conditioning on both left and right context in all layers. The pretrained BERT model can be fine-tuned with just one additional output layer (in many cases) to create state-of-the-art models. This model can use for a wide range of NLP tasks, such as question answering and language inference, and so on without substantial task-specific architecture modification.</p>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="https://camo.githubusercontent.com/c95655003ee4b649902d5cdc8832168b558fb52060e4f1f4111a2d5ca88b3a86/68747470733a2f2f7265732e636c6f7564696e6172792e636f6d2f6d33687264616466692f696d6167652f75706c6f61642f76313539353135383939312f6b6167676c652f626572745f696e707574735f7738726974682e706e67"><img src="https://camo.githubusercontent.com/c95655003ee4b649902d5cdc8832168b558fb52060e4f1f4111a2d5ca88b3a86/68747470733a2f2f7265732e636c6f7564696e6172792e636f6d2f6d33687264616466692f696d6167652f75706c6f61642f76313539353135383939312f6b6167676c652f626572745f696e707574735f7738726974682e706e67" alt="BERT INPUTS" data-canonical-src="https://res.cloudinary.com/m3hrdadfi/image/upload/v1595158991/kaggle/bert_inputs_w8rith.png" style="max-width: 100%;"></a></p>
<p dir="auto">As you may know, the BERT model input is a combination of 3 embeddings.</p>
<ul dir="auto">
<li>Token embeddings: WordPiece token vocabulary (WordPiece is another word segmentation algorithm, similar to BPE)</li>
<li>Segment embeddings: for pair sentences [A-B] marked as <math-renderer class="js-inline-math" style="display: inline" data-static-url="https://github.githubassets.com/static">$E_A$</math-renderer> or <math-renderer class="js-inline-math" style="display: inline" data-static-url="https://github.githubassets.com/static">$E_B$</math-renderer> mean that it belongs to the first sentence or the second one.</li>
<li>Position embeddings: specify the position of words in a sentence</li>
</ul>
<p dir="auto"><br><br>
Before going more further into code, let us introduce ParsBERT.
<br></p>
<h2 dir="auto"><a id="user-content-parsbert" class="anchor" aria-hidden="true" href="#parsbert"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>ParsBERT</h2>
<p dir="auto">ParsBERT is a monolingual language model based on Google's BERT architecture. This model is pre-trained on large Persian corpora with various writing styles from numerous subjects (e.g., scientific, novels, news, ...) with more than 3.9M documents, 73M sentences, and 1.3B words. For more information about ParsBERT, please check out the article: <a href="https://link.springer.com/article/10.1007/s11063-021-10528-4" rel="nofollow">DOI: 10.1007/s11063-021-10528-4</a></p>
<br>
So, now you have a little understanding of BERT in total, we need to know how to use ParsBERT in our project. In this assignment, you will implement a fine-tuned model on the Sentiment Analysis task for PyTorch. Good Luck!
<p dir="auto"><br><br></p>
<p dir="auto"><strong>Setup</strong></p>
<ul dir="auto">
<li>Download assignment3.ipynb to obtain the assignment jupyter notebook.</li>
<li>Go to <a href="https://colab.research.google.com/" rel="nofollow">https://colab.research.google.com/</a>.</li>
<li>Switch to Upload tab, choose assignment3.ipynb and click upload.</li>
<li>Now You’re ready to go.</li>
  </body>
</html>

