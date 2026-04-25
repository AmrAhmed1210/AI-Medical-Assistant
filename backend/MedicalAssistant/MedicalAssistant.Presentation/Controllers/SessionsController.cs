using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Shared.DTOs.SessionDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;
using System.Text.RegularExpressions;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api/sessions")]
    [Authorize]
    public class SessionsController : ControllerBase
    {
        private readonly ISessionService _sessionService;
        private readonly IMessageService _messageService;
        private readonly INotificationService _notificationService;
        private readonly IDoctorService _doctorService;
        private readonly IUnitOfWork _unitOfWork;

        public SessionsController(
            ISessionService sessionService,
            IMessageService messageService,
            INotificationService notificationService,
            IDoctorService doctorService,
            IUnitOfWork unitOfWork)
        {
            _sessionService = sessionService;
            _messageService = messageService;
            _notificationService = notificationService;
            _doctorService = doctorService;
            _unitOfWork = unitOfWork;
        }

        // GET /api/sessions
        public async Task<IActionResult> GetMySessions()
        {
            var role = User.FindFirstValue(ClaimTypes.Role) ?? string.Empty;
            IEnumerable<SessionDto> sessions;
            var currentUserId = GetCurrentUserId();
            if (currentUserId <= 0) return Unauthorized();

            if (role.Equals("Admin", StringComparison.OrdinalIgnoreCase))
            {
                // Admin sees all support sessions + any other sessions they might need to moderate
                var allSessions = await _sessionService.GetAllSessionsAsync();
                sessions = allSessions.OrderByDescending(s => s.UpdatedAt ?? s.CreatedAt);
            }
            else if (role.Equals("Doctor", StringComparison.OrdinalIgnoreCase))
            {
                var doctorId = await GetCurrentDoctorProfileIdAsync(currentUserId);
                if (!doctorId.HasValue) return Ok(Enumerable.Empty<SessionDto>());

                var (allSessions, _) = await _sessionService.GetPaginatedSessionsAsync(1, 2000);
                var doctorSessions = allSessions
                    .Where(s => ExtractDoctorIdFromTitle(s.Title) == doctorId.Value)
                    .ToList();

                await ReplaceSessionTitlesWithPatientNamesAsync(doctorSessions);
                sessions = doctorSessions;
            }
            else
            {
                sessions = await _sessionService.GetSessionsByUserIdAsync(currentUserId);
            }

            return Ok(sessions);
        }

        // GET /api/sessions/{id}
        [HttpGet("{id:int}")]
        public async Task<IActionResult> GetSession(int id)
        {
            var session = await _sessionService.GetSessionByIdAsync(id);
            if (session == null)
                return NotFound();

            if (!await CanAccessSessionAsync(session))
                return Forbid();

            var role = User.FindFirstValue(ClaimTypes.Role) ?? string.Empty;
            if (role.Equals("Doctor", StringComparison.OrdinalIgnoreCase))
            {
                await ReplaceSessionTitleWithPatientNameAsync(session);
            }

            session.Messages = (await _messageService.GetMessagesForSessionAsync(id)).ToList();
            return Ok(session);
        }

        // DELETE /api/sessions/{id}
        [HttpDelete("{id:int}")]
        public async Task<IActionResult> DeleteSession(int id)
        {
            var session = await _sessionService.GetSessionByIdAsync(id);
            if (session == null) return NotFound();
            if (!await CanAccessSessionAsync(session)) return Forbid();

            var ok = await _sessionService.DeleteSessionAsync(id);
            if (!ok) return NotFound();
            return NoContent();
        }

        // POST /api/sessions
        [HttpPost]
        public async Task<IActionResult> StartSession([FromBody] StartSessionRequest req)
        {
            var role = User.FindFirstValue(ClaimTypes.Role) ?? string.Empty;
            if (!role.Equals("Patient", StringComparison.OrdinalIgnoreCase))
                return Forbid();

            var userId = GetCurrentUserId();
            if (userId <= 0)
                return Unauthorized();

            if (req == null || req.DoctorId <= 0)
                return BadRequest(new { message = "doctorId is required" });

            var doctorExists = await _unitOfWork.Repository<Doctor>()
                .GetByIdAsync(req.DoctorId);
            if (doctorExists == null)
                return NotFound(new { message = "Doctor not found" });

            var patientIdClaim = User.FindFirst("PatientId")?.Value;
            if (!int.TryParse(patientIdClaim, out var patientId))
            {
                var patient = (await _unitOfWork.Repository<Patient>().FindAsync(p => p.UserId == userId)).FirstOrDefault();
                patientId = patient?.Id ?? 0;
            }

            var sessionTitle = BuildSessionTitle(patientId, req.DoctorId);
            var existing = (await _sessionService.GetSessionsByUserIdAsync(userId))
                .FirstOrDefault(s => string.Equals(s.Title, sessionTitle, StringComparison.OrdinalIgnoreCase));

            var session = existing ?? await _sessionService.CreateSessionAsync(userId, sessionTitle, "DoctorChat");

            if (!string.IsNullOrWhiteSpace(req.InitialMessage))
            {
                await _messageService.SendMessageAsync(session.Id, userId, "user", req.InitialMessage.Trim());
                
                // Notify doctor about the new message
                var patientUser = await _unitOfWork.Repository<User>().GetByIdAsync(userId);
                if (patientUser != null)
                {
                    var doctorUser = await _unitOfWork.Repository<Doctor>().GetByIdAsync(req.DoctorId);
                    if (doctorUser != null && doctorUser.UserId.HasValue)
                    {
                        var docUser = await _unitOfWork.Repository<User>().GetByIdAsync(doctorUser.UserId.Value);
                        if (docUser != null)
                        {
                            await _notificationService.NotifyDoctorNewMessage(
                                docUser.Email,
                                patientUser.FullName ?? "Patient",
                                req.InitialMessage.Trim(),
                                session.Id,
                                doctorUser.Id);
                        }
                    }
                }
            }

            return CreatedAtAction(nameof(GetSession), new { id = session.Id }, session);
        }

        // POST /api/sessions/support
        [HttpPost("support")]
        public async Task<IActionResult> StartSupportSession([FromBody] StartSupportRequest req)
        {
            var userId = GetCurrentUserId();
            if (userId <= 0) return Unauthorized();

            var role = User.FindFirstValue(ClaimTypes.Role) ?? "User";
            var userName = User.FindFirstValue(ClaimTypes.Name);
            
            if (string.IsNullOrWhiteSpace(userName))
            {
                var user = await _unitOfWork.Repository<User>().GetByIdAsync(userId);
                userName = user?.FullName ?? "User #" + userId;
            }

            var sessionTitle = $"Support | {role}: {userName}";

            // Find existing non-deleted support session for this user
            var existing = (await _sessionService.GetSessionsByUserIdAsync(userId))
                .FirstOrDefault(s => s.Type == "SupportChat" && !s.IsDeleted);

            var session = existing ?? await _sessionService.CreateSessionAsync(userId, sessionTitle, "SupportChat");

            if (!string.IsNullOrWhiteSpace(req.Message))
            {
                var senderRole = role.Equals("Doctor", StringComparison.OrdinalIgnoreCase) ? "doctor" : "user";
                await _messageService.SendMessageAsync(session.Id, userId, senderRole, req.Message.Trim());
            }

            return Ok(session);
        }

        // POST /api/sessions/{id}/message
        [HttpPost("{id:int}/message")]
        [HttpPost("{id:int}/messages")]
        public async Task<IActionResult> SendMessage(int id, [FromBody] SendMessageRequest req)
        {
            if (string.IsNullOrWhiteSpace(req.Content))
                return BadRequest(new { message = "Message content is required" });

            var userId = GetCurrentUserId();
            if (userId <= 0)
                return Unauthorized();

            var session = await _sessionService.GetSessionByIdAsync(id);
            if (session == null) return NotFound();
            if (!await CanAccessSessionAsync(session)) return Forbid();

            var role = User.FindFirstValue(ClaimTypes.Role) ?? string.Empty;
            var senderRole = role.Equals("Doctor", StringComparison.OrdinalIgnoreCase) ? "doctor" : 
                             role.Equals("Admin", StringComparison.OrdinalIgnoreCase) ? "admin" : "user";
            
            var msg = await _messageService.SendMessageAsync(id, userId, senderRole, req.Content.Trim(), req.MessageType, req.AttachmentUrl, req.FileName);

            if (senderRole == "doctor")
            {
                var patientUser = await _unitOfWork.Repository<User>()
                    .GetByIdAsync(session.UserId);
                var doctorProfile = await _doctorService.GetProfileAsync(userId);
                var patientEmail = patientUser?.Email?.Trim();
                if (!string.IsNullOrWhiteSpace(patientEmail) && doctorProfile != null)
                {
                    await _notificationService.NotifyNewMessage(
                        patientEmail,
                        doctorProfile.FullName ?? "Your Doctor",
                        req.Content.Trim(),
                        id,
                        doctorProfile.Id);
                }
            }
            else if (senderRole == "user")
            {
                var patientIdClaim = User.FindFirst("PatientId")?.Value;
                if (!int.TryParse(patientIdClaim, out var patientId))
                {
                    var p = (await _unitOfWork.Repository<Patient>().FindAsync(pa => pa.UserId == userId)).FirstOrDefault();
                    patientId = p?.Id ?? 0;
                }

                var patientUser = await _unitOfWork.Repository<User>().GetByIdAsync(userId);
                var doctorId = ExtractDoctorIdFromTitle(session.Title);
                if (doctorId.HasValue && patientUser != null)
                {
                    var doctorUser = await _unitOfWork.Repository<Doctor>().GetByIdAsync(doctorId.Value);
                    if (doctorUser != null && doctorUser.UserId.HasValue)
                    {
                        var docUser = await _unitOfWork.Repository<User>().GetByIdAsync(doctorUser.UserId.Value);
                        if (docUser != null)
                        {
                            await _notificationService.NotifyDoctorNewMessage(
                                docUser.Email,
                                patientUser.FullName ?? "Patient",
                                req.Content.Trim(),
                                id,
                                doctorUser.Id,
                                patientId,
                                patientUser.PhotoUrl);
                        }
                    }
                }
            }

            if (session.Type == "SupportChat")
            {
                var senderUser = await _unitOfWork.Repository<User>().GetByIdAsync(userId);
                var senderName = senderUser?.FullName ?? "User";
                
                if (senderRole == "admin")
                {
                    // Notify the user that admin replied
                    var targetUser = await _unitOfWork.Repository<User>().GetByIdAsync(session.UserId);
                    if (targetUser != null && !string.IsNullOrWhiteSpace(targetUser.Email))
                    {
                        await _notificationService.NotifyNewMessage(targetUser.Email, "Support Team", req.Content.Trim(), id);
                    }
                }
                else
                {
                    // Notify admin (or system log) about user message
                    await _notificationService.NotifyDoctorNewMessage("Admin", senderName, req.Content.Trim(), id);
                }
            }

            return Ok(msg);
        }

        private int GetCurrentUserId()
        {
            var userIdClaim = User.FindFirstValue(ClaimTypes.NameIdentifier)
                           ?? User.FindFirstValue("UserId")
                           ?? User.FindFirstValue("sub")
                           ?? User.FindFirstValue("PatientId");

            return int.TryParse(userIdClaim, out var userId) ? userId : 0;
        }

        private async Task<int?> GetCurrentDoctorProfileIdAsync(int currentUserId)
        {
            var claimDoctorId = User.FindFirstValue("DoctorId");
            if (int.TryParse(claimDoctorId, out var doctorId) && doctorId > 0)
                return doctorId;

            var doctor = (await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.DoctorsModule.Doctor>()
                .FindAsync(d => d.UserId == currentUserId))
                .FirstOrDefault();

            return doctor?.Id;
        }

        private async Task<bool> CanAccessSessionAsync(SessionDetailDto session)
        {
            var role = User.FindFirstValue(ClaimTypes.Role) ?? string.Empty;
            var currentUserId = GetCurrentUserId();
            if (currentUserId <= 0) return false;

            // Robust Admin check
            if (role.Equals("Admin", StringComparison.OrdinalIgnoreCase) || User.IsInRole("Admin"))
                return true;

            if (role.Equals("Doctor", StringComparison.OrdinalIgnoreCase))
            {
                var doctorId = await GetCurrentDoctorProfileIdAsync(currentUserId);
                if (!doctorId.HasValue) return false;
                return ExtractDoctorIdFromTitle(session.Title) == doctorId.Value;
            }

            return session.UserId == currentUserId;
        }

        private static string BuildSessionTitle(int patientId, int doctorId)
            => $"chat|p:{patientId}|d:{doctorId}|";

        private static int? ExtractDoctorIdFromTitle(string? title)
        {
            if (string.IsNullOrWhiteSpace(title)) return null;
            var match = Regex.Match(title, @"\|d:(\d+)\|", RegexOptions.IgnoreCase);
            return match.Success && int.TryParse(match.Groups[1].Value, out var id) ? id : null;
        }

        private static int? ExtractPatientIdFromTitle(string? title)
        {
            if (string.IsNullOrWhiteSpace(title)) return null;
            var match = Regex.Match(title, @"\|p:(\d+)\|", RegexOptions.IgnoreCase);
            return match.Success && int.TryParse(match.Groups[1].Value, out var id) ? id : null;
        }

        private async Task ReplaceSessionTitlesWithPatientNamesAsync(List<SessionDto> sessions)
        {
            foreach (var session in sessions)
            {
                var patientId = ExtractPatientIdFromTitle(session.Title);
                
                // 🛠️ Auto-Fix: If title is missing patient ID or is 0, resolve it via UserId
                if (patientId == null || patientId <= 0)
                {
                    var patient = (await _unitOfWork.Repository<Patient>().FindAsync(p => p.UserId == session.UserId)).FirstOrDefault();
                    if (patient != null)
                    {
                        patientId = patient.Id;
                        var docId = ExtractDoctorIdFromTitle(session.Title) ?? 0;
                        session.Title = BuildSessionTitle(patient.Id, docId);
                        
                        // Update in DB too to fix it permanently
                        var dbSession = await _unitOfWork.Sessions.GetByIdAsync(session.Id);
                        if (dbSession != null)
                        {
                            dbSession.Title = session.Title;
                            _unitOfWork.Sessions.Update(dbSession);
                        }
                    }
                }

                if (patientId.HasValue && patientId > 0)
                {
                    var name = "";
                    var photoUrl = "";

                    var patient = await _unitOfWork.Repository<Patient>().GetByIdAsync(patientId.Value);
                    if (patient != null)
                    {
                        name = patient.FullName;
                        photoUrl = patient.ImageUrl;
                        
                        if (string.IsNullOrWhiteSpace(name) || string.IsNullOrWhiteSpace(photoUrl))
                        {
                            var user = await _unitOfWork.Repository<User>().GetByIdAsync(patient.UserId ?? 0);
                            if (user != null)
                            {
                                if (string.IsNullOrWhiteSpace(name)) name = user.FullName;
                                if (string.IsNullOrWhiteSpace(photoUrl)) photoUrl = user.PhotoUrl;
                            }
                        }
                    }
                    else
                    {
                        var user = await _unitOfWork.Repository<User>().GetByIdAsync(session.UserId);
                        if (user != null)
                        {
                            name = user.FullName;
                            photoUrl = user.PhotoUrl;
                        }
                    }

                    if (!string.IsNullOrWhiteSpace(name))
                    {
                        session.Title = name.Trim();
                        session.PatientName = name.Trim();
                    }
                    session.PatientPhotoUrl = photoUrl;
                }
            }
            
            await _unitOfWork.SaveChangesAsync();
        }

        private async Task ReplaceSessionTitleWithPatientNameAsync(SessionDetailDto session)
        {
            var patientId = ExtractPatientIdFromTitle(session.Title);
            if (!patientId.HasValue || patientId <= 0) return;

            var name = "";
            var photoUrl = "";

            var patient = await _unitOfWork.Repository<Patient>().GetByIdAsync(patientId.Value);
            if (patient != null)
            {
                name = patient.FullName;
                photoUrl = patient.ImageUrl;
                
                if (string.IsNullOrWhiteSpace(name) || string.IsNullOrWhiteSpace(photoUrl))
                {
                    var user = await _unitOfWork.Repository<User>().GetByIdAsync(patient.UserId ?? 0);
                    if (user != null)
                    {
                        if (string.IsNullOrWhiteSpace(name)) name = user.FullName;
                        if (string.IsNullOrWhiteSpace(photoUrl)) photoUrl = user.PhotoUrl;
                    }
                }
            }
            else
            {
                var user = await _unitOfWork.Repository<User>().GetByIdAsync(session.UserId);
                if (user != null)
                {
                    name = user.FullName;
                    photoUrl = user.PhotoUrl;
                }
            }

            if (!string.IsNullOrWhiteSpace(name))
            {
                session.Title = name.Trim();
                session.PatientName = name.Trim();
            }
            session.PatientPhotoUrl = photoUrl;
        }
    }

    public class StartSessionRequest
    {
        public int DoctorId { get; set; }
        public string? InitialMessage { get; set; }
    }

    public class SendMessageRequest
    {
        public string Content { get; set; } = string.Empty;
        public string MessageType { get; set; } = "text";
        public string? AttachmentUrl { get; set; }
        public string? FileName { get; set; }
    }

    public class StartSupportRequest
    {
        public string? Message { get; set; }
    }
}