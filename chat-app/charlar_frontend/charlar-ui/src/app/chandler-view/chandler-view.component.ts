import { ChatServiceService } from './../_service/chat-service.service';
import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-chandler-view',
  templateUrl: './chandler-view.component.html',
  styleUrls: ['./chandler-view.component.scss']
})
export class ChandlerViewComponent implements OnInit {


  public chat_info :any= [];
  public titleValue;
  public noteValue;
  public msg:string = "Share your thoughts now 💡";

  constructor(public chatServiceService: ChatServiceService) { }

  ngOnInit(): void {
    this.loadChats()
  }
  
  loadChats(){
    this.chatServiceService.fetch_nodes()
          .subscribe((data)=>{
            this.chat_info =  data;
          },error  => {
            console.log(error)            
            });

  }
  deleteNote(noteId){
    this.chatServiceService.deleteThoughts(noteId)
          .subscribe((data)=>{
            this.loadChats();
          },error  => {
            console.log(error)            
            });
  }
  pushNote(){
    console.log("value here",this.titleValue,this.noteValue)
    let payloads = {
      'title':this.titleValue,
      'thought':this.noteValue
    }
    this.chatServiceService.pushThoughts(payloads)
          .subscribe((data)=>{            
            this.loadChats();
            this.msg = data['message'];
            
            
          },error  => {
            console.log(error)
            this.msg =  error.message
            
            });

  }

}
