// Initialize animations when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Hide loader after page loads
    const loader = document.getElementById('loader');
    anime({
        targets: loader,
        opacity: 0,
        duration: 1000,
        easing: 'easeOutQuad',
        complete: () => {
            loader.style.display = 'none';
        }
    });

    // Animate navigation items
    anime({
        targets: '.nav-items li',
        translateY: [20, 0],
        opacity: [0, 1],
        duration: 800,
        delay: anime.stagger(100),
        easing: 'easeOutQuad'
    });

    // Animate hero content
    anime({
        targets: ['.title', '.subtitle', '.cta-button'],
        translateY: [50, 0],
        opacity: [0, 1],
        duration: 1200,
        delay: anime.stagger(200),
        easing: 'easeOutExpo'
    });

    // Initialize form animations
    initFormAnimations();
});

// Form animations
function initFormAnimations() {
    const formGroups = document.querySelectorAll('.form-group');
    
    // Animate form groups when they come into view
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                anime({
                    targets: entry.target,
                    translateY: [20, 0],
                    opacity: [0, 1],
                    duration: 800,
                    easing: 'easeOutQuad'
                });
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1
    });

    formGroups.forEach(group => observer.observe(group));
}

// Animate form submission
function animateFormSubmission() {
    anime({
        targets: '.form-container',
        scale: [1, 0.98],
        duration: 200,
        easing: 'easeInOutQuad',
        complete: () => {
            anime({
                targets: '.form-container',
                scale: [0.98, 1],
                duration: 400,
                easing: 'easeOutElastic(1, .5)'
            });
        }
    });
}

// Animate results appearance
function showResults() {
    const resultsContainer = document.getElementById('resultsContainer');
    resultsContainer.style.display = 'block';

    anime({
        targets: '.results-container',
        translateY: [50, 0],
        opacity: [0, 1],
        duration: 1000,
        easing: 'easeOutExpo'
    });

    // Animate prediction cards
    anime({
        targets: ['.prediction-card', '.alternatives-grid > div'],
        scale: [0.9, 1],
        opacity: [0, 1],
        duration: 800,
        delay: anime.stagger(100),
        easing: 'easeOutElastic(1, .5)'
    });
}

// Smooth scroll to sections
function scrollToPredict() {
    const predictSection = document.getElementById('predict');
    window.scrollTo({
        top: predictSection.offsetTop - 80,
        behavior: 'smooth'
    });
}

// Hobby animation
function addHobbyAnimation(hobbyElement) {
    anime({
        targets: hobbyElement,
        translateX: [-50, 0],
        opacity: [0, 1],
        duration: 500,
        easing: 'easeOutQuad'
    });
}

// Remove hobby animation
function removeHobbyAnimation(hobbyElement, callback) {
    anime({
        targets: hobbyElement,
        translateX: [0, 50],
        opacity: [1, 0],
        duration: 500,
        easing: 'easeInQuad',
        complete: callback
    });
}

// Input range value animations
function initRangeAnimations() {
    const ranges = document.querySelectorAll('input[type="range"]');
    ranges.forEach(range => {
        range.addEventListener('input', (e) => {
            const value = e.target.nextElementSibling;
            anime({
                targets: value,
                innerHTML: [parseFloat(value.innerHTML), e.target.value],
                round: 1,
                duration: 200,
                easing: 'linear'
            });
        });
    });
}

// Initialize all animations
initRangeAnimations();
