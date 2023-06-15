import { Injectable } from '@angular/core';
import { Router, CanActivate, ActivatedRouteSnapshot, RouterStateSnapshot } from '@angular/router';
import { AuthenticationServiceService } from '../_service/authentication-service.service';


@Injectable({
  providedIn: 'root'
})
export class AuthGuardService {

  constructor(
    public _router:Router,
    public _authService:AuthenticationServiceService) { 

      // constructor
    }

    canActivate(route: ActivatedRouteSnapshot, state: RouterStateSnapshot) {
      const currentUser = this._authService.currentUserValue;
      if (currentUser) {
          // authorised so return true
          return true;
      }

      // not logged in so redirect to login page with the return url
      this._router.navigate(['/login'], { queryParams: { returnUrl: state.url }});
      return false;
  }
}
