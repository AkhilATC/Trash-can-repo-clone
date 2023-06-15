import { Injectable } from '@angular/core';
import {
  HttpRequest,
  HttpHandler,
  HttpEvent,
  HttpInterceptor
} from '@angular/common/http';
import { Observable } from 'rxjs';
import { AuthenticationServiceService } from '../_service/authentication-service.service';
import { ChatServiceService } from '../_service/chat-service.service';

@Injectable()
export class JwtInterceptor implements HttpInterceptor {

  constructor(public _authenticationService:AuthenticationServiceService,public chatService:ChatServiceService ) {}

  intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    let currentUser = this._authenticationService.currentUserValue;
        console.log(currentUser)
        if (currentUser && currentUser) {
            request = request.clone({
                setHeaders: { 
                    Authorization: `Bearer ${currentUser.token}`
                }
            });
        }
    return next.handle(request);
  }
}
