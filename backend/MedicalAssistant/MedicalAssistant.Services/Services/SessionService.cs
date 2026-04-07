using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.SessionsModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.SessionDTOs;

namespace MedicalAssistant.Services.Services
{
    public class SessionService : ISessionService
    {
        private readonly IUnitOfWork _unitOfWork;
        private readonly ISessionRepository _repo;

        public SessionService(IUnitOfWork unitOfWork)
        {
            _unitOfWork = unitOfWork;
            _repo = (ISessionRepository)_unitOfWork.GetType().GetProperty("Sessions")?.GetValue(_unitOfWork);
        }

        public async Task<SessionDto> CreateSessionAsync(int userId, string? title = null)
        {
            var session = new Session
            {
                UserId = userId,
                Title = title,
                CreatedAt = DateTime.UtcNow
            };
            await _repo.AddAsync(session);
            await _unitOfWork.SaveChangesAsync();
            return Map(session);
        }

        public async Task<SessionDetailDto?> GetSessionByIdAsync(int id)
        {
            var s = await _repo.GetByIdAsync(id);
            if (s == null) return null;

            // Messages + analysis can be plugged here later
            return new SessionDetailDto
            {
                Id = s.Id,
                UserId = s.UserId,
                Title = s.Title,
                CreatedAt = s.CreatedAt,
                UpdatedAt = s.UpdatedAt,
                MessageCount = 0,
                UrgencyLevel = s.UrgencyLevel,
                Messages = new(),
                AnalysisResult = null,
            };
        }

        public async Task<IEnumerable<SessionDto>> GetSessionsByUserIdAsync(int userId)
        {
            var items = await _repo.GetByUserIdAsync(userId);
            return items.Select(Map);
        }

        public async Task<(IEnumerable<SessionDto> Items, int TotalCount)> GetPaginatedSessionsAsync(int pageNumber, int pageSize)
        {
            var (items, total) = await _repo.GetPaginatedAsync(pageNumber, pageSize);
            return (items.Select(Map), total);
        }

        public async Task<bool> DeleteSessionAsync(int id)
        {
            var s = await _repo.GetByIdAsync(id);
            if (s == null) return false;
            s.IsDeleted = true;
            _repo.Update(s);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<SessionDto?> UpdateLastMessageTimeAsync(int id)
        {
            var s = await _repo.GetByIdAsync(id);
            if (s == null) return null;
            s.UpdatedAt = DateTime.UtcNow;
            _repo.Update(s);
            await _unitOfWork.SaveChangesAsync();
            return Map(s);
        }

        private static SessionDto Map(Session s) => new SessionDto
        {
            Id = s.Id,
            UserId = s.UserId,
            Title = s.Title,
            UrgencyLevel = s.UrgencyLevel,
            IsDeleted = s.IsDeleted,
            CreatedAt = s.CreatedAt,
            UpdatedAt = s.UpdatedAt,
            MessageCount = 0,
        };
    }
}
