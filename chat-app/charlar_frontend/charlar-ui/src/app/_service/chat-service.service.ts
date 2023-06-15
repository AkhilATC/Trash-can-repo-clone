import { environment } from './../../environments/environment';
import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ChatServiceService {

  constructor(private http: HttpClient) { }

  public fetch_nodes(){
    return this.http.get(`${environment.chat_api_uri}/fetch_posts`);
  }
  public pushThoughts(payloads){
    return this.http.post(`${environment.chat_api_uri}/publish_my_note`,payloads);
  }
  public deleteThoughts(id){
    return this.http.delete(`${environment.chat_api_uri}/delete/${id}`);
  }
  public register(data){  
    var payload = {
      "name":data['username'],
      "first_name":data['firstName'],
      "last_name":data['lastName'],
      "password":data['password']
    }
    
    return this.http.post(`${environment.auth_sign_in_url}/sign_up`,payload);
  }
}
