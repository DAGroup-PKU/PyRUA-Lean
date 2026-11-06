const button = document.querySelector('.citation-copy');
const code = document.querySelector('#citation-code');
let resetTimer;
button?.addEventListener('click', async () => {
  clearTimeout(resetTimer);
  try {
    await navigator.clipboard.writeText(code.textContent.trim() + '\n');
    button.textContent = 'Copied!';
  } catch {
    button.textContent = 'Copy failed';
  }
  resetTimer = setTimeout(() => { button.textContent = 'Copy'; }, 2000);
});
