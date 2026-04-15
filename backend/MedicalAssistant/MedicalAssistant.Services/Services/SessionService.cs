using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.SessionsModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.SessionDTOs;

namespace MedicalAssistant.Services.Services
{
    public class SessionService : ISessionService
    {
        private readonly IUnitOfWork _unitOfWork;

        public SessionService(IUnitOfWork unitOfWork)
        {
            _unitOfWork = unitOfWork;
        }

        public async Task<SessionDto> CreateSessionAsync(int userId, string? title = null)
        {
            var session = new Session
            {
                UserId    = userId,
                Title     = title,
                CreatedAt = DateTime.UtcNow
            };

            await _unitOfWork.Sessions.AddAsync(session);
            await _unitOfWork.SaveChangesAsync();
            return Map(session);
        }

        public async Task<SessionDetailDto?> GetSessionByIdAsync(int id)
        {
            var s = await _unitOfWork.Sessions.GetByIdAsync(id);
            if (s == null) return null;

            return new SessionDetailDto
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
            };
        }

        public async Task<IEnumerable<SessionDto>> GetSessionsByUserIdAsync(int userId)
        {
            var items = await _unitOfWork.Sessions.GetByUserIdAsync(userId);
            return items.Select(Map);
        }

        public async Task<(IEnumerable<SessionDto> Items, int TotalCount)> GetPaginatedSessionsAsync(int pageNumber, int pageSize)
        {
            var (items, total) = await _unitOfWork.Sessions.GetPaginatedAsync(pageNumber, pageSize);
            return (items.Select(Map), total);
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
            var s = await _unitOfWork.Sessions.GetByIdAsync(id);
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
            IsDeleted    = s.IsDeleted,
            CreatedAt    = s.CreatedAt,
            UpdatedAt    = s.UpdatedAt,
            MessageCount = 0,
        };
    }
}