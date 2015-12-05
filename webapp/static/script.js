function showmenu(i) {	
	i += 3;
	document.getElementById("sidebar").style.left = "" + i + "%";
	setTimeout(function() {
		if(i < 0) {
	 		showmenu(i);
		};
	}, 20);
};

function hidemenu(i) {	
	i-=3;
	document.getElementById("sidebar").style.left = "" + i + "%";
	setTimeout(function() {
		if(i > -30){
			hidemenu(i);
		};
	}, 20);
};