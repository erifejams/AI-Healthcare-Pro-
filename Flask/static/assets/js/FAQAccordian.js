
//this is to get the accordian section, so that when the user clicks an event will happen e.g it will expand 
const accordion = document.getElementsByClassName('accordianBox');

for (i=0; i<accordion.length; i++) {
  accordion[i].addEventListener('click', function () {
    this.classList.toggle('active')
  })
}