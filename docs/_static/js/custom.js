
function addGithubButton() {
    const div = `
        <div class="github-repo">
            <a 
                class="github-button"
                href="https://github.com/huggingface/datasets" data-size="large" data-show-count="true" aria-label="Star huggingface/datasets on GitHub">
                Star
            </a>
        </div>
    `;
    document.querySelector(".wy-side-nav-search .icon-home").insertAdjacentHTML('afterend', div);
}
function onLoad() {
    addGithubButton();
}

window.addEventListener("load", onLoad);