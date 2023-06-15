import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { User } from './_models/user';
import { AuthenticationServiceService } from './_service/authentication-service.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'charlar-ui';
  currentUser: User;

  constructor(
    private router: Router,
    private authenticationService: AuthenticationServiceService
) {
    this.authenticationService.currentUser.subscribe(x => this.currentUser = x);
}

logout() {
    this.authenticationService.logout();
    this.router.navigate(['/login']);
}
}
