using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.SessionDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;
using System.Threading.Tasks;
using System.Collections.Generic; // Required for List
using System;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api/sessions")]
    [Authorize]
    public class SessionsController : ControllerBase
    {
        private readonly ISessionService _sessionService;
        private readonly IMessageService _messageService;

        public SessionsController(ISessionService sessionService, IMessageService messageService)
        {
            _sessionService = sessionService;
            _messageService = messageService;
        }

        [HttpGet]
        public async Task<IActionResult> GetSessions()
        {
            // user id ?? ??? JWT (??? ????? ??? int)
            var userIdClaim = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (!int.TryParse(userIdClaim, out var userId))
                return Unauthorized();

            var sessions = await _sessionService.GetSessionsByUserIdAsync(userId);
            return Ok(sessions);
        }

        [HttpGet("{id:int}")]
        public async Task<IActionResult> GetSession(int id)
        {
            var session = await _sessionService.GetSessionByIdAsync(id);
            if (session == null)
                return NotFound();

            // ???? ??????? ?? ??????
            session.Messages = (await _messageService.GetMessagesForSessionAsync(id)).ToList();
            return Ok(session);
        }

        [HttpDelete("{id:int}")]
        public async Task<IActionResult> DeleteSession(int id)
        {
            var ok = await _sessionService.DeleteSessionAsync(id);
            if (!ok)
                return NotFound();

            return NoContent();
        }

        [HttpPost]
        public async Task<IActionResult> StartSession([FromBody] CreateMessageDto? body)
        {
            var userIdClaim = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (!int.TryParse(userIdClaim, out var userId))
                return Unauthorized();

            var session = await _sessionService.CreateSessionAsync(userId, "Chest pain consultation");

            // ?? ??? ??? ????? ?? ??????? ??????
            if (body != null && !string.IsNullOrWhiteSpace(body.Content))
            {
                await _messageService.SendMessageAsync(session.Id, userId, "user", body.Content);
            }

            return CreatedAtAction(nameof(GetSession), new { id = session.Id }, session);
        }

        [HttpPost("{id:int}/message")]
        public async Task<IActionResult> SendMessage(int id, [FromBody] CreateMessageDto dto)
        {
            if (string.IsNullOrWhiteSpace(dto.Content))
                return BadRequest(new { message = "Message content is required" });

            var userIdClaim = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (!int.TryParse(userIdClaim, out var userId))
                return Unauthorized();

            var msg = await _messageService.SendMessageAsync(id, userId, "user", dto.Content);
            return Ok(msg);
        }
    }
}
