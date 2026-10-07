"""Optional presentation keyboard controls; no data access or transformations."""
import streamlit.components.v1 as components


def keyboard_navigation():
    components.html('''<script>
    const parentWindow = window.parent;
    const doc = parentWindow.document;
    if (parentWindow.sjdPresentationKeys) {
      doc.removeEventListener('keydown', parentWindow.sjdPresentationKeys);
    }
    parentWindow.sjdPresentationKeys = (event) => {
      if (['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName) || event.target.isContentEditable) return;
      const label = event.key === 'ArrowRight' ? 'Next →' : event.key === 'ArrowLeft' ? '← Previous' : null;
      if (!label) return;
      const button = [...doc.querySelectorAll('button')].find(button => button.textContent.trim() === label);
      if (button && !button.disabled) { event.preventDefault(); button.click(); }
    };
    doc.addEventListener('keydown', parentWindow.sjdPresentationKeys);
    </script>''',height=0)
