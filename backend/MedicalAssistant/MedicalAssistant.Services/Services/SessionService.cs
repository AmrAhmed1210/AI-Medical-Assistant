using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.SessionsModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.SessionDTOs;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant.Services.Services
{
    public class SessionService : ISessionService
    {
        private readonly IUnitOfWork _unitOfWork;

        public SessionService(IUnitOfWork unitOfWork)
        {
            _unitOfWork = unitOfWork;
        }

        public async Task<SessionDto> CreateSessionAsync(int userId, string? title = null, string type = "AI")
        {
            var session = new Session
            {
                UserId    = userId,
                Title     = title,
                Type      = type,
                CreatedAt = DateTime.UtcNow
            };

            await _unitOfWork.Sessions.AddAsync(session);
            await _unitOfWork.SaveChangesAsync();
            
            // Reload to get User navigation property
            var reloaded = await _unitOfWork.Repository<Session>()
                .Query()
                .Include(s => s.User)
                .FirstOrDefaultAsync(s => s.Id == session.Id);

            return Map(reloaded ?? session);
        }

        public async Task<SessionDetailDto?> GetSessionByIdAsync(int id)
        {
            var s = await _unitOfWork.Repository<Session>()
                .Query()
                .Include(s => s.User)
                .FirstOrDefaultAsync(s => s.Id == id);
                
            if (s == null) return null;

            var dto = new SessionDetailDto
            {
                Id             = s.Id,
                UserId         = s.UserId,
                Title          = s.Title,
                CreatedAt      = s.CreatedAt,
                UpdatedAt      = s.UpdatedAt,
                MessageCount   = 0,
                UrgencyLevel   = s.UrgencyLevel,
                Messages       = new(),
                AnalysisResult = null,
                PatientName    = s.User?.FullName,
                PatientPhotoUrl = s.User?.PhotoUrl
            };

            return dto;
        }

        public async Task<IEnumerable<SessionDto>> GetSessionsByUserIdAsync(int userId)
        {
            var items = await _unitOfWork.Repository<Session>()
                .Query()
                .Include(s => s.User)
                .Where(s => s.UserId == userId && !s.IsDeleted)
                .OrderByDescending(s => s.UpdatedAt ?? s.CreatedAt)
                .ToListAsync();

            var dtos = new List<SessionDto>();

            foreach (var item in items)
            {
                var dto = Map(item);
                var lastMsg = (await _unitOfWork.Messages.GetBySessionIdAsync(item.Id))
                    .OrderByDescending(m => m.Timestamp)
                    .FirstOrDefault();

                if (lastMsg != null)
                {
                    dto.LastMessage = lastMsg.Content;
                    dto.LastMessageAt = lastMsg.Timestamp;
                    dto.UpdatedAt = lastMsg.Timestamp;
                }
                dtos.Add(dto);
            }

            return dtos;
        }

        public async Task<IEnumerable<SessionDto>> GetAllSessionsAsync()
        {
            var items = await _unitOfWork.Repository<Session>()
                .Query()
                .Include(s => s.User)
                .Where(s => !s.IsDeleted)
                .OrderByDescending(s => s.UpdatedAt ?? s.CreatedAt)
                .ToListAsync();

            var dtos = new List<SessionDto>();

            foreach (var item in items)
            {
                var dto = Map(item);
                var lastMsg = (await _unitOfWork.Messages.GetBySessionIdAsync(item.Id))
                    .OrderByDescending(m => m.Timestamp)
                    .FirstOrDefault();

                if (lastMsg != null)
                {
                    dto.LastMessage = lastMsg.Content;
                    dto.LastMessageAt = lastMsg.Timestamp;
                    dto.UpdatedAt = lastMsg.Timestamp;
                }
                dtos.Add(dto);
            }

            return dtos;
        }

        public async Task<(IEnumerable<SessionDto> Items, int TotalCount)> GetPaginatedSessionsAsync(int pageNumber, int pageSize)
        {
            var query = _unitOfWork.Repository<Session>()
                .Query()
                .Include(s => s.User)
                .Where(s => !s.IsDeleted);

            var total = await query.CountAsync();
            var items = await query
                .OrderByDescending(s => s.UpdatedAt ?? s.CreatedAt)
                .Skip((pageNumber - 1) * pageSize)
                .Take(pageSize)
                .ToListAsync();

            var dtos = new List<SessionDto>();

            foreach (var item in items)
            {
                var dto = Map(item);
                var lastMsg = (await _unitOfWork.Messages.GetBySessionIdAsync(item.Id))
                    .OrderByDescending(m => m.Timestamp)
                    .FirstOrDefault();

                if (lastMsg != null)
                {
                    dto.LastMessage = lastMsg.Content;
                    dto.LastMessageAt = lastMsg.Timestamp;
                }
                dtos.Add(dto);
            }

            return (dtos, total);
        }

        public async Task<bool> DeleteSessionAsync(int id)
        {
            var s = await _unitOfWork.Sessions.GetByIdAsync(id);
            if (s == null) return false;

            s.IsDeleted = true;
            _unitOfWork.Sessions.Update(s);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<SessionDto?> UpdateLastMessageTimeAsync(int id)
        {
            var s = await _unitOfWork.Repository<Session>()
                .GetByIdAsync(id, s => s.User);
                
            if (s == null) return null;

            s.UpdatedAt = DateTime.UtcNow;
            _unitOfWork.Sessions.Update(s);
            await _unitOfWork.SaveChangesAsync();
            return Map(s);
        }

        private static SessionDto Map(Session s) => new SessionDto
        {
            Id           = s.Id,
            UserId       = s.UserId,
            Title        = s.Title,
            UrgencyLevel = s.UrgencyLevel,
            Type         = s.Type,
            IsDeleted    = s.IsDeleted,
            CreatedAt    = s.CreatedAt,
            UpdatedAt    = s.UpdatedAt ?? s.CreatedAt,
            MessageCount = 0,
            LastMessage = null,
            LastMessageAt = null,
            PatientName = s.User?.FullName,
            PatientPhotoUrl = s.User?.PhotoUrl
        };
    }
}