const sidebar = document.getElementById('toggle_button');
const nav = document.getElementById('nav');
const name = document.getElementById('name-user');
const toggleIcon = document.getElementById('toggle_icon');
const dropdowns = document.querySelectorAll('.dropdown');

let position = -280;

sidebar.addEventListener('click', () => {
    if (position === -280) {
        position = 0;
        toggleIcon.classList.add('rotated');
    } else {
        position = -280;
        toggleIcon.classList.remove('rotated');
    }
    nav.style.left = `${position}px`;
});

document.addEventListener('DOMContentLoaded', async () => {
    const nameCache = localStorage.getItem('user');
    if (nameCache) {
        name.textContent = nameCache;
    }

    await getMenuOptions();
});

/*dropbutton.addEventListener('click', (e) => {
    e.preventDefault();
    const submenu = dropbutton.nextElementSibling;
    submenu.style.display = submenu.style.display === 'block' ? 'none' : 'block';
  });*/

dropdowns.forEach(dropdown => {
    const button = dropdown.querySelector('.drop-btn');
    button.addEventListener('click', () => {
      // Cierra todos los dropdowns excepto el actual
      dropdowns.forEach(d => {
        if (d !== dropdown) {
          d.classList.remove('open');
        }
      });

      dropdown.classList.toggle('open');
    });
  });

  // Cerrar si se hace clic fuera
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.dropdown')) {
      dropdowns.forEach(d => d.classList.remove('open'));
    }
  });

async function viewCloseSession(){
    localStorage.removeItem('token');
    localStorage.removeItem('id');
    localStorage.removeItem('user');
    localStorage.removeItem('paths');
    localStorage.removeItem('frames');
    localStorage.removeItem('rol')
    window.location.href = "/login";
}

async function getMenuOptions(){
    let request = await fetch('/get_menu_options',{
        method: 'GET'
    });

    const response = await request.json();
    const manageUsersUrl = "{{ url_for('manage_users') }}";

    if (localStorage.getItem('rol') != "1") {
        document.getElementById('user_control').remove();
    }
}